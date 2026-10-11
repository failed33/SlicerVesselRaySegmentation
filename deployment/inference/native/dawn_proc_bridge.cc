// Compile this source into the host's webgpu_dawn target, so the returned
// procedure table belongs to the implementation that owns the Python buffers.
// Returned pointers borrow that library's lifetime; the caller keeps it loaded.
#include <cstdint>
#include <cstring>

#include "dawn/native/DawnNative.h"

#if defined(_WIN32)
#define DAWN_ORT_EXPORT __declspec(dllexport)
#else
#define DAWN_ORT_EXPORT __attribute__((visibility("default")))
#endif

extern "C" DAWN_ORT_EXPORT auto dawn_ort_get_native_procs() noexcept -> const DawnProcTable* {
    return &dawn::native::GetProcs();
}

extern "C" DAWN_ORT_EXPORT auto dawn_ort_get_proc_version() noexcept -> const uint8_t* {
    return &dawn::native::GetProcs().version[0];
}

extern "C" DAWN_ORT_EXPORT auto dawn_ort_get_proc_address(const char* name) noexcept -> WGPUProc {
    return dawn::native::GetProcs().getProcAddress({name, std::strlen(name)});
}
