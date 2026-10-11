#include "projections.hpp"
#include "task_trace.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ggml-alloc.h>
#include <ggml-vulkan.h>
#include <iostream>
#include <stdexcept>
namespace strata::vulkan {
using Clock = std::chrono::steady_clock;
std::vector<float> decode_activations(std::span<const float> input) {
  if (input.size() % 32)
    throw std::invalid_argument(
        "Q8_1 activation width must be a multiple of 32");
  std::vector<float> output(input.size());
  for (size_t first = 0; first < input.size(); first += 32) {
    float maximum = 0;
    for (size_t i = first; i < first + 32; ++i) {
      if (!std::isfinite(input[i]))
        throw std::invalid_argument("Nonfinite Vulkan activation");
      maximum = std::max(maximum, std::abs(input[i]));
    }
    const float scale = std::min(maximum / 127.f, 65504.f);
    const float stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
    for (size_t i = first; i < first + 32; ++i)
      output[i] =
          maximum == 0 || stored == 0
              ? 0
              : std::clamp(std::round(input[i] / scale), -127.f, 127.f) *
                    stored;
  }
  return output;
}
Projections::Projections(int device, size_t budget)
    : budget_(budget), device_(device) {
  // GGML's FP32 batched shader can still convert operands to FP16. Start the
  // experimental runner with the strict path; this does not affect HIP/CUDA.
#ifdef _WIN32
  _putenv_s("GGML_VK_DISABLE_F16", "1");
  _putenv_s("GGML_VK_DISABLE_COOPMAT", "1");
  _putenv_s("GGML_VK_DISABLE_COOPMAT2", "1");
#else
  if (setenv("GGML_VK_DISABLE_F16", "1", 1) ||
      setenv("GGML_VK_DISABLE_COOPMAT", "1", 1) ||
      setenv("GGML_VK_DISABLE_COOPMAT2", "1", 1))
    throw std::runtime_error(
        "Cannot select strict Vulkan projection precision");
#endif
  if (device < 0 || device >= ggml_backend_vk_get_device_count())
    throw std::invalid_argument("Vulkan device index unavailable");
  backend_ = ggml_backend_vk_init(device);
  if (!backend_)
    throw std::runtime_error("Vulkan backend initialization failed");
  std::cerr << "VULKAN_PROJECTIONS device=" << device_name()
            << " budget_MiB=" << budget_ / 1048576 << '\n';
}
Projections::~Projections() {
  for (auto &[n, w] : weights_) {
    if (w.plan_buffer)
      ggml_backend_buffer_free(w.plan_buffer);
    if (w.plan_context)
      ggml_free(w.plan_context);
    ggml_backend_buffer_free(w.buffer);
    ggml_free(w.context);
  }
  if (backend_)
    ggml_backend_free(backend_);
}
std::string Projections::device_name() const {
  char text[512]{};
  ggml_backend_vk_get_device_description(device_, text, sizeof(text));
  return text;
}
std::vector<float> Projections::mat(const std::string &name,
                                    const strata::core::ArtifactTensor &t,
                                    std::span<const float> x, int heads,
                                    int tokens) {
  if (t.tensor->shape.size() < 2 || t.tensor->shape.size() > 3)
    throw std::invalid_argument("Vulkan matrix rank invalid: " + name);
  if ((heads == 1 && t.tensor->shape.size() != 2) ||
      (heads > 1 &&
       (t.tensor->shape.size() != 3 || t.tensor->shape[2] != uint64_t(heads))))
    throw std::invalid_argument("Vulkan matrix head geometry invalid: " + name);
  return product(name, ggml_type(t.tensor->type), t.data(), t.bytes,
                 int(t.tensor->shape[0]), int(t.tensor->shape[1]), x, tokens,
                 heads);
}
std::vector<float>
Projections::synthetic(const std::string &key, ggml_type type,
                       std::span<const uint8_t> bytes, int in, int out,
                       std::span<const float> x, int tokens, int heads) {
  return product(key, type, bytes.data(), bytes.size(), in, out, x, tokens,
                 heads);
}
std::vector<float> Projections::product(const std::string &key, ggml_type type,
                                        const void *source, size_t bytes,
                                        int in, int out,
                                        std::span<const float> x, int tokens,
                                        int heads) {
  if (in <= 0 || out <= 0 || heads <= 0 || tokens <= 0 || tokens > 2048 ||
      x.size() != size_t(in) * tokens * heads)
    throw std::invalid_argument("Vulkan projection geometry invalid: " + key);
  auto found = weights_.find(key);
  if (found == weights_.end()) {
    if (bytes > budget_ - resident_)
      throw std::runtime_error("Vulkan resident projection budget exceeded: " +
                               key);
    Weight w;
    w.context = ggml_init({ggml_tensor_overhead() * 2 + 4096, nullptr, true});
    if (!w.context)
      throw std::bad_alloc();
    try {
      w.tensor = ggml_new_tensor_3d(w.context, type, in, out, heads);
      ggml_set_name(w.tensor, key.c_str());
      if (ggml_nbytes(w.tensor) != bytes)
        throw std::invalid_argument("Vulkan weight byte count differs: " + key);
      w.buffer = ggml_backend_alloc_ctx_tensors(w.context, backend_);
      if (!w.buffer)
        throw std::bad_alloc();
      w.allocated = ggml_backend_buffer_get_size(w.buffer);
      if (w.allocated > budget_ - resident_)
        throw std::runtime_error(
            "Vulkan padded weight allocation exceeds budget: " + key);
      auto start = Clock::now();
      ggml_backend_tensor_set(w.tensor, source, 0, bytes);
      transfer_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      w.bytes = bytes;
      if (heads == 1 && (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0)) {
        const size_t blocks = size_t(in) / 32;
        const size_t block_bytes = ggml_type_size(type);
        const float offset = type == GGML_TYPE_Q4_0 ? -8.f : -16.f;
        w.affine_bias.resize(size_t(out) * blocks);
        for (size_t i = 0; i < w.affine_bias.size(); ++i) {
          ggml_fp16_t scale;
          std::memcpy(&scale,
                      static_cast<const uint8_t *>(source) + i * block_bytes,
                      sizeof(scale));
          w.affine_bias[i] = offset * ggml_fp16_to_fp32(scale);
        }
      }
      found = weights_.emplace(key, w).first;
      resident_ += w.allocated;
    } catch (...) {
      if (w.buffer)
        ggml_backend_buffer_free(w.buffer);
      ggml_free(w.context);
      throw;
    }
  }
  auto &w = found->second;
  if (w.tensor->type != type || w.tensor->ne[0] != in ||
      w.tensor->ne[1] != out || w.tensor->ne[2] != heads || w.bytes != bytes)
    throw std::invalid_argument("Vulkan cached projection geometry changed: " +
                                key);
  if (w.tokens != tokens) {
    if (w.plan_buffer)
      ggml_backend_buffer_free(w.plan_buffer);
    if (w.plan_context)
      ggml_free(w.plan_context);
    resident_ -= w.plan_bytes;
    w.plan_buffer = nullptr;
    w.plan_context = nullptr;
    w.plan_bytes = 0;
    w.tokens = 0;
    ggml_context *ctx =
        ggml_init({ggml_tensor_overhead() * 8 +
                       ggml_graph_overhead_custom(32, false) + 4096,
                   nullptr, true});
    if (!ctx)
      throw std::bad_alloc();
    ggml_backend_buffer_t buffer = nullptr;
    try {
      auto *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, tokens, heads);
      auto *output = ggml_mul_mat(ctx, w.tensor, input);
      if (!ggml_prec_set_acc(output, GGML_PREC_F32))
        throw std::runtime_error(
            "Vulkan projection cannot request FP32 accumulation");
      if (!ggml_backend_supports_op(backend_, output))
        throw std::runtime_error("Vulkan projection operation unsupported: " +
                                 key);
      auto *graph = ggml_new_graph_custom(ctx, 32, false);
      ggml_build_forward_expand(graph, output);
      buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_);
      if (!buffer)
        throw std::bad_alloc();
      const auto plan_bytes = ggml_backend_buffer_get_size(buffer);
      if (plan_bytes > budget_ - resident_)
        throw std::runtime_error(
            "Vulkan projection workspace exceeds budget: " + key);
      w.plan_context = ctx;
      w.plan_buffer = buffer;
      w.input = input;
      w.output = output;
      w.graph = graph;
      w.tokens = tokens;
      w.plan_bytes = plan_bytes;
      resident_ += plan_bytes;
    } catch (...) {
      if (buffer)
        ggml_backend_buffer_free(buffer);
      ggml_free(ctx);
      throw;
    }
  }
  auto start = Clock::now();
  auto quantized = ggml_is_quantized(type) && heads == 1 ? decode_activations(x)
                                                         : std::vector<float>{};
  ggml_backend_tensor_set(w.input,
                          quantized.empty() ? x.data() : quantized.data(), 0,
                          x.size_bytes());
  transfer_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  start = Clock::now();
  operation_trace("projection_begin", key);
  auto status = ggml_backend_graph_compute(backend_, w.graph);
  compute_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (status != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Vulkan projection execution failed: " + key);
  operation_trace("projection_complete", key);
  std::vector<float> y(size_t(out) * tokens * heads);
  start = Clock::now();
  ggml_backend_tensor_get(w.output, y.data(), 0, y.size() * 4);
  transfer_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  if (!w.affine_bias.empty()) {
    // Native Q4_0/Q5_0 dots use the original-input sum stored in Q8_1, not
    // the sum of reconstructed quantized activations. Keep that correction.
    start = Clock::now();
    const int blocks = in / 32;
    std::vector<float> delta(size_t(tokens) * blocks);
    for (int t = 0; t < tokens; ++t)
      for (int b = 0; b < blocks; ++b) {
        std::array<float, 32> sums{};
        float reconstructed = 0;
        const size_t first = size_t(t) * in + b * 32;
        for (int i = 0; i < 32; ++i) {
          sums[i] = x[first + i];
          reconstructed += quantized[first + i];
        }
        for (int offset = 16; offset; offset /= 2) {
          const auto old = sums;
          for (int i = 0; i < 32; ++i)
            sums[i] = old[i] + old[i ^ offset];
        }
        const float original = ggml_fp16_to_fp32(
            ggml_fp32_to_fp16(std::clamp(sums[0], -65504.f, 65504.f)));
        delta[size_t(t) * blocks + b] = original - reconstructed;
      }
    for (int t = 0; t < tokens; ++t)
      for (int r = 0; r < out; ++r) {
        float correction = 0;
        for (int b = 0; b < blocks; ++b)
          correction += w.affine_bias[size_t(r) * blocks + b] *
                        delta[size_t(t) * blocks + b];
        y[size_t(t) * out + r] += correction;
      }
    affine_cpu_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  for (float v : y)
    if (!std::isfinite(v))
      throw std::runtime_error("Vulkan projection returned nonfinite values: " +
                               key);
  ++calls;
  return y;
}
} // namespace strata::vulkan
