cmake_minimum_required(VERSION 3.28)

set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Native asset configuration")
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      14.0
      CACHE STRING "Oldest supported macOS")
endif()
project(dawn_ort_native LANGUAGES CXX)
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)
include(FetchContent)
include(ExternalProject)
include(pins.cmake)
set_property(DIRECTORY PROPERTY EP_BASE "${CMAKE_CURRENT_BINARY_DIR}/external")

# Trust the selected release's published checksums; CMake caches its metadata.
FetchContent_Declare(
  dawn_release
  URL "https://api.github.com/repos/google/dawn/releases/tags/${DAWN_RELEASE}"
  DOWNLOAD_NAME release.json
  DOWNLOAD_NO_EXTRACT TRUE
  TLS_VERIFY TRUE)
FetchContent_MakeAvailable(dawn_release)
file(READ "${dawn_release_SOURCE_DIR}/release.json" _release_metadata)
string(JSON _asset_count LENGTH "${_release_metadata}" assets)
math(EXPR _last_asset "${_asset_count} - 1")

# Download a named Dawn archive after checking its published release checksum.
# Return the FetchContent source directory to the calling directory.
function(dawn_archive name archive)
  foreach(index RANGE 0 ${_last_asset} 1)
    string(JSON asset GET "${_release_metadata}" assets ${index})
    string(JSON asset_name GET "${asset}" name)
    if(asset_name STREQUAL archive)
      string(JSON digest GET "${asset}" digest)
      string(REPLACE "sha256:" "SHA256=" checksum "${digest}")
      FetchContent_Declare(
        ${name}
        URL "${DAWN_RELEASE_URL}/${archive}"
        URL_HASH "${checksum}"
        TLS_VERIFY TRUE)
      FetchContent_MakeAvailable(${name})
      # FetchContent requires this generated variable name.
      set(${name}_SOURCE_DIR # cmake-lint: disable=C0103
          "${${name}_SOURCE_DIR}"
          PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "Dawn release ${DAWN_RELEASE} has no asset ${archive}")
endfunction()
dawn_archive(dawn_headers "${DAWN_HEADERS_ARCHIVE}")

# Exact commits need an explicit depth-one fetch; GIT_SHALLOW cannot pin a SHA.
# The provider and optional source-built backend use this same checkout.
ExternalProject_Add(
  dawn_source
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/sources/dawn"
  DOWNLOAD_COMMAND "${GIT_EXECUTABLE}" init <SOURCE_DIR>
  COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> fetch --depth 1
          https://github.com/google/dawn.git "${DAWN_COMMIT}"
  COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> checkout --detach FETCH_HEAD
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND "")
set(_dawn_dependencies dawn_source)
if(DAWN_BUILD_FROM_SOURCE)
  set(_dawn_backend_build "${CMAKE_CURRENT_BINARY_DIR}/dawn-backend")
  set(_dawn_library "${_dawn_backend_build}/lib/libwebgpu_dawn.a")
  ExternalProject_Add(
    dawn_backend
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/sources/dawn"
    BINARY_DIR "${_dawn_backend_build}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release"
               "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
               "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
               "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
               "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${_dawn_backend_build}/lib"
               "-DBUILD_SHARED_LIBS=OFF"
               "-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC"
               "-DDAWN_SUPPORTS_CXX_MODULES=OFF"
               "-DDAWN_FETCH_DEPENDENCIES=ON"
               "-DDAWN_ENABLE_VULKAN=ON"
               "-DDAWN_ENABLE_NULL=OFF"
               "-DDAWN_ENABLE_DESKTOP_GL=OFF"
               "-DDAWN_ENABLE_OPENGLES=OFF"
               "-DDAWN_USE_X11=OFF"
               "-DDAWN_USE_WAYLAND=OFF"
               "-DDAWN_USE_GLFW=OFF"
               "-DDAWN_BUILD_SAMPLES=OFF"
               "-DDAWN_BUILD_TESTS=OFF"
               "-DDAWN_BUILD_PROTOBUF=OFF"
               "-DTINT_BUILD_CMD_TOOLS=OFF"
               "-DTINT_BUILD_TESTS=OFF"
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target webgpu_dawn
    USES_TERMINAL_BUILD TRUE
    BUILD_BYPRODUCTS "${_dawn_library}"
    INSTALL_COMMAND ""
    DEPENDS dawn_source)
  # Fetch the backend dependencies once before ORT configures the same source.
  list(APPEND _dawn_dependencies dawn_backend)
else()
  dawn_archive(dawn_binary "${DAWN_ARCHIVE}")
  set(_dawn_library "${dawn_binary_SOURCE_DIR}/${DAWN_STATIC_LIBRARY}")
endif()

set(_dawn_include "${dawn_headers_SOURCE_DIR}/include")
set(_artifacts "${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>")
add_library(dawn_native_archive STATIC IMPORTED)
set_property(TARGET dawn_native_archive PROPERTY IMPORTED_LOCATION
                                                 "${_dawn_library}")
if(DAWN_BUILD_FROM_SOURCE)
  add_dependencies(dawn_native_archive dawn_backend)
endif()
add_library(webgpu_dawn SHARED dawn_proc_bridge.cc)
target_compile_features(webgpu_dawn PRIVATE cxx_std_20)
target_include_directories(webgpu_dawn PRIVATE "${_dawn_include}")
target_link_libraries(
  webgpu_dawn
  PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,dawn_native_archive>"
          ${DAWN_SYSTEM_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(
  webgpu_dawn
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_artifacts}"
             RUNTIME_OUTPUT_DIRECTORY "${_artifacts}"
             CXX_SCAN_FOR_MODULES OFF)
if(APPLE)
  set_target_properties(webgpu_dawn PROPERTIES INSTALL_NAME_DIR "@rpath"
                                               BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# The libclang wheel has no builtin C headers. Use the selected Linux
# toolchain's headers when parsing the C API, including stddef.h.
set(_bindings_environment)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  foreach(include_directory IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    if(EXISTS "${include_directory}/stddef.h")
      set(_bindings_include_path "${include_directory}")
      if(NOT "$ENV{C_INCLUDE_PATH}" STREQUAL "")
        string(APPEND _bindings_include_path ":$ENV{C_INCLUDE_PATH}")
      endif()
      set(_bindings_environment "${CMAKE_COMMAND}" -E env
                                "C_INCLUDE_PATH=${_bindings_include_path}")
      break()
    endif()
  endforeach()
endif()

add_custom_command(
  OUTPUT "${_artifacts}/webgpu.py"
  COMMAND
    ${_bindings_environment} "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_bindings.py"
    "$<TARGET_FILE:webgpu_dawn>" "${_dawn_include}" "${_artifacts}/webgpu.py"
    "${DAWN_COMMIT}" "libclang${CMAKE_SHARED_LIBRARY_SUFFIX}"
  DEPENDS webgpu_dawn "${CMAKE_CURRENT_SOURCE_DIR}/generate_bindings.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/pins.cmake"
          "${_dawn_include}/dawn/webgpu.h"
  COMMENT "Generate ctypes bindings for pinned Dawn headers"
  VERBATIM)
add_custom_target(
  bindings ALL
  DEPENDS "${_artifacts}/webgpu.py"
  COMMENT "Build pinned Dawn ctypes bindings")

set(_provider_build "${CMAKE_CURRENT_BINARY_DIR}/provider")
string(CONCAT _provider_name "${CMAKE_SHARED_LIBRARY_PREFIX}"
              "onnxruntime_providers_webgpu" "${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_provider_library "${_provider_build}/${_provider_name}")
set(_ort_runtime_patch
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/pr32688-runtime.patch")
set(_ort_wgsl_patch
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/wgsl-optional-variable-prerequisite.patch"
)
ExternalProject_Add(
  ort_provider
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/sources/ort"
  BINARY_DIR "${_provider_build}"
  DOWNLOAD_COMMAND "${GIT_EXECUTABLE}" init <SOURCE_DIR>
  COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> fetch --depth 1
          https://github.com/microsoft/onnxruntime.git "${ORT_COMMIT}"
  COMMAND "${GIT_EXECUTABLE}" -C <SOURCE_DIR> checkout --detach FETCH_HEAD
  PATCH_COMMAND
    "${CMAKE_COMMAND}" "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "-DSOURCE_DIR=<SOURCE_DIR>" "-DEXPECTED_COMMIT=${ORT_COMMIT}"
    "-DRUNTIME_PATCH=${_ort_runtime_patch}" "-DWGSL_PATCH=${_ort_wgsl_patch}"
    "-DPATCH_STATE_FILE=<TMP_DIR>/native-ort-patches.sha256" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/apply_patches.cmake"
  UPDATE_COMMAND ""
  SOURCE_SUBDIR cmake
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS
    "--compile-no-warning-as-error"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
    "-DPython_EXECUTABLE=${Python3_EXECUTABLE}"
    "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${CMAKE_CURRENT_SOURCE_DIR}/ort.cmake"
    "-DDAWN_ORT_BUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target
                onnxruntime_providers_webgpu
  USES_TERMINAL_BUILD TRUE
  BUILD_BYPRODUCTS "${_provider_library}"
  INSTALL_COMMAND ""
  DEPENDS ${_dawn_dependencies})
ExternalProject_Add_StepDependencies(
  ort_provider patch "${CMAKE_CURRENT_SOURCE_DIR}/apply_patches.cmake"
  "${_ort_runtime_patch}" "${_ort_wgsl_patch}")
ExternalProject_Add_StepDependencies(ort_provider configure
                                     "${CMAKE_CURRENT_SOURCE_DIR}/ort.cmake")

# Verify the provider's actual generated headers instead of generating a second
# copy.
set(_provider_headers
    "${CMAKE_CURRENT_BINARY_DIR}/deps/dawn-build/gen/include/dawn")
ExternalProject_Add_Step(
  ort_provider verify_headers
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${_provider_headers}/webgpu.h"
          "${_dawn_include}/dawn/webgpu.h"
  COMMAND
    "${CMAKE_COMMAND}" -E compare_files "${_provider_headers}/dawn_proc_table.h"
    "${_dawn_include}/dawn/dawn_proc_table.h"
  DEPENDS "${_dawn_include}/dawn/webgpu.h"
          "${_dawn_include}/dawn/dawn_proc_table.h"
  DEPENDEES build
  DEPENDERS install)

install(
  TARGETS webgpu_dawn
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .)
install(FILES "${_artifacts}/webgpu.py" "${_provider_library}" DESTINATION .)
