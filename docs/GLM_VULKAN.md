# Experimental GLM Vulkan runner

`STRATA_ENABLE_VULKAN=ON` builds `strata-glm-vulkan` without CUDA, HIP or ROCm.
The first runner targets the GLM-5.3f NoPE, four-stream REAP50 GGUF model on one
Vulkan GPU plus system RAM. Validation hardware is an RX 9070 XT, Ryzen 3950X
in ECO mode, X570S AERO G and 64 GiB DDR4-2666. Other AMD cards and Windows
execution have not been tested. Vulkan avoids ROCm's architecture selection;
it still requires a working Vulkan driver and the compute/memory features
checked at startup.

The GPU runs dense and per-head projections, mHC mixing, KDA recurrence and
sparse MLA attention. KDA state and MLA caches remain resident between tokens.
The existing CPU pool runs routed experts. Normalization, convolution,
SwiGLU, index pooling and selection currently run on the CPU. Inputs and
outputs cross a host boundary between projection and mixer calls.

This is a standalone experimental runner. Setup and the API server do not yet
select it. Prefill is sequential, with no resident GPU expert tier, speculative
decoding, CUDA graphs or multi-GPU execution. The benchmark rejects unsupported
settings. It is not a claim that the optimized HIP runner has been fully ported.

## Build

On Linux, install a C++20 compiler, CMake, Ninja, Vulkan development headers,
`glslc` and SPIR-V headers. Mesa RADV is the tested Vulkan driver; ROCm is not
needed by this executable. Use a separate build directory:

```sh
cmake -S . -B build-vulkan -G Ninja \
  -DSTRATA_ENABLE_VULKAN=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF
cmake --build build-vulkan --target strata-glm-vulkan glm_vulkan_parity -j 8
```

The build uses the same pinned GGML checkout as the CPU expert path. An existing
checkout can be supplied with `-DSTRATA_GGML_DIR=<llama.cpp checkout>`. For a
Vulkan SDK outside the system prefix, supply `CMAKE_PREFIX_PATH` and
`Vulkan_GLSLC_EXECUTABLE`. The custom SPIR-V shaders are embedded in the binary;
no shader files or ROCm libraries are needed at runtime.

The tested workspace build used glslc 2023.8 and SPIR-V headers 1.4.309.
That compiler did not enable GGML's integer-dot-product shader feature. This
limits the measured implementation; the results do not establish Vulkan's
maximum possible speed on RDNA4.

## Validate and run

Run `build-vulkan/vulkan/glm_vulkan_parity` first. It compares FP32 and quantized
projections with CPU oracles, including Q4_0/Q5_0's original-input-sum correction,
checks repeated cached projections, five KDA steps and retained state, mHC,
sparse MLA and rejection of invalid selected rows. Projection and custom-kernel
checks use a relative L2 limit of `2e-5`; they do not prove full-model quality.

```sh
build-vulkan/vulkan/strata-glm-vulkan \
  --model /path/to/GLM-5.3-Flash-REAP50-Q23-assembled.gguf \
  --prompt-file /path/to/prompt.ids --output /tmp/vulkan-result.json \
  --context 4096 --generate 64 --trials 3 --threads 12 --gpu-budget-mib 15360
```

Prompts contain comma-separated integer token IDs. Generated IDs go to stdout;
measurements go to stderr and the requested JSON file. `--stop-ids` accepts a
comma-separated list, including the model's end-of-turn tokens. Existing output
files are rejected. Three trials include an untimed prefill warmup; cold expert
page misses can still affect the first generated response, so inspect all trials
and I/O evidence.

Quantized two-dimensional projections reproduce Q8_1 activation scale/rounding.
Q4_0/Q5_0 also retain the original-input-sum term used by the HIP/CUDA dot
product. Per-head MLA products keep FP32 activations. FP16 operand conversion
and cooperative-matrix paths are disabled in the initial strict projection
profile, with FP32 accumulation requested explicitly.

Custom working buffers prefer host-cached coherent memory so the CPU does not
read result vectors through uncached mapped VRAM. Persistent state prefers
host-visible device-local memory and can use coherent system memory when a
visible-VRAM heap cannot satisfy an allocation. That fallback was not exercised
on the validation card. Allocations, tensor geometry, selected rows
and append order are checked. Each custom dispatch waits for its own fence,
with a 60-second bound; a failed fence exits without freeing in-flight buffers.

## Fault evidence and comparisons

The [guarded diagnostic workflow](GLM_GPU_DIAGNOSTICS.md) also accepts benchmark
configs with `backend: "vulkan"`. It retains the 60 GiB RAM/zero-swap limit,
physical GPU headroom, owned-service cleanup, PCI identity checks and timestamped
telemetry. Vulkan runs additionally hash their backend and shader sources.
Engine phase tracing is optional and must be disabled for performance timings.

`STRATA_GLM_STATE_TRACE_DIR` saves the first token's residual after every layer.
It refuses to overwrite files. The HIP reference rejects this diagnostic in
pipelined execution, where synchronization can wait for host work not yet
started. Layer dumps are diagnostic artifacts, not inputs for simulator fitting.

The October 12 comparison uses the same Q23 model and token IDs, 12 CPU workers,
no expert tier, sequential prefill, three trials and the same resource limits.
The initial Vulkan implementation matched 192 generated tokens but was slower
than that HIP control. Earlier smoke tests also found a full-distribution
mismatch; both failures and subsequent corrections are retained. See the
[measurement record](benchmarks/glm_vulkan_9070xt_20261012.json) for final numbers,
quality checks and the remaining limitations. Full held-out coding and
perplexity qualification are separate from these short comparisons.

The corrected warm results were:

| Backend | Sequential prefill tok/s | Decode tok/s | Sampled peak global VRAM MiB |
| --- | ---: | ---: | ---: |
| HIP / ROCm 10.1, no expert tier | 8.641 | 8.534 | 6,723 |
| Vulkan, corrected staging and affine dots | 3.784 | 3.739 | 5,487 |

Vulkan reached 43.8% of this matched HIP control's decode rate, saved 1,236 MiB
of sampled global VRAM, and improved 2.45 times over its first warm baseline
(1.527 tok/s). All 192 generated IDs matched across three trials. This is a
comparison against the sequential, no-tier control, not the optimized HIP tier.
The frozen prompt IDs are in `bench/fixtures/glm53_vulkan_prefix32.ids`.

On a separate 53-token C++ prime prompt, both backends passed 400,538 compiled
oracle cases, including unsigned overflow/boundary inputs. HIP generated 129
tokens and Vulkan 138. These traced cold runs are correctness checks rather
than performance measurements. They do not replace the full held-out suite.

The corrected 32-token prefill still had KL 0.0624 from HIP to Vulkan, above the
0.01 numerical gate. The remaining discrepancy is unresolved. Keep the runner
experimental; these measurements do not qualify it as the default coding path.
No GPU/PCI hardware fault was recorded during this campaign. Older AMD cards,
batched prefill, long-context quality and speculative decoding remain untested.

Vulkan engine traces also bracket each named projection and custom shader with
its context position and layer. A failed GPU fence exits without freeing
in-flight buffers. Read the last persisted operation after a reboot using the
diagnostic status tool; an interrupted run is never a passing measurement.
