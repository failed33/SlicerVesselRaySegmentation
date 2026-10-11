// The ROCm 7.1.1 C wrapper does not expose the native ONNX operator inventory.
// Keep its C++ types in this translation unit, separate from C-wrapper types.
#include <migraphx/onnx.hpp>
#include <string>
#include <vector>

namespace mgx_ep {

std::vector<std::string> GetRocm711OnnxOperators() {
    return migraphx::get_onnx_operators();
}

}  // namespace mgx_ep
