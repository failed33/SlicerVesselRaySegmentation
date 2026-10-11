# Use the build worker's interpreter for ORT's generators. CMake and ORT require
# these exact upstream variable names.
set(Python3_EXECUTABLE "${Python_EXECUTABLE}" # cmake-lint: disable=C0103
)
foreach(provider_option IN ITEMS BUILD_SHARED_LIB USE_WEBGPU
                                 USE_EP_API_ADAPTERS USE_EXTERNAL_DAWN)
  set(onnxruntime_${provider_option} # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
endforeach()
set(onnxruntime_BUILD_UNIT_TESTS # cmake-lint: disable=C0103
    OFF
    CACHE BOOL "" FORCE)
set(onnxruntime_USE_TELEMETRY # cmake-lint: disable=C0103
    OFF
    CACHE BOOL "" FORCE)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
set(FETCHCONTENT_BASE_DIR "${DAWN_ORT_BUILD_ROOT}/deps")
set(FETCHCONTENT_SOURCE_DIR_DAWN "${DAWN_ORT_BUILD_ROOT}/sources/dawn")
# These consumers use Dawn's C API and ordinary headers. GCC 13 passes Dawn's
# module syntax probe but lacks the GCC 14 scanner required by CMake's FILE_SET.
set(DAWN_SUPPORTS_CXX_MODULES
    OFF
    CACHE BOOL "Build Dawn without its unused C++ module interface" FORCE)
set(FORCE_RESET_OSX_DEPLOYMENT_TARGET
    OFF
    CACHE BOOL "" FORCE)
