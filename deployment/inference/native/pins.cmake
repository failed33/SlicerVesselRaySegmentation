# Keep binary/source identities together; archive checksums come from the
# release.
set(DAWN_COMMIT b0dde004842f35ae3ee86706724318a4086c9a80)
set(ORT_COMMIT f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7)
set(DAWN_RELEASE v20260818.211311)
set(DAWN_RELEASE_URL
    https://github.com/google/dawn/releases/download/${DAWN_RELEASE})
set(DAWN_HEADERS_ARCHIVE dawn-headers-${DAWN_COMMIT}.tar.gz)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _dawn_architecture)
if(_dawn_architecture STREQUAL "amd64")
  set(_dawn_architecture x86_64)
elseif(_dawn_architecture STREQUAL "aarch64")
  set(_dawn_architecture arm64)
endif()
set(DAWN_BUILD_FROM_SOURCE OFF)

if(APPLE)
  set(DAWN_ARCHIVE dawn-apple-${DAWN_COMMIT}.xcframework.tar.gz)
  set(DAWN_STATIC_LIBRARY macos-arm64_x86_64/libwebgpu_dawn.a)
  set(DAWN_ARCHITECTURES arm64 x86_64)
  set(DAWN_SYSTEM_LIBRARIES
      "-framework CoreFoundation"
      "-framework Foundation"
      "-framework IOSurface"
      "-framework QuartzCore"
      "-framework Cocoa"
      "-framework IOKit"
      "-framework Metal")
elseif(WIN32)
  set(DAWN_ARCHIVE Dawn-${DAWN_COMMIT}-windows-latest-Release.tar.gz)
  set(DAWN_STATIC_LIBRARY lib/webgpu_dawn.lib)
  set(DAWN_ARCHITECTURES x86_64)
  set(DAWN_SYSTEM_LIBRARIES d3d12 dxgi dxguid user32 advapi32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(DAWN_ARCHITECTURES x86_64 arm64)
  if(_dawn_architecture STREQUAL "arm64")
    # The pinned release has no Linux ARM64 backend archive.
    set(DAWN_BUILD_FROM_SOURCE ON)
  else()
    set(DAWN_ARCHIVE Dawn-${DAWN_COMMIT}-ubuntu-latest-Release.tar.gz)
    set(DAWN_STATIC_LIBRARY lib64/libwebgpu_dawn.a)
  endif()
else()
  message(FATAL_ERROR "No pinned Dawn archive for ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT _dawn_architecture IN_LIST DAWN_ARCHITECTURES)
  message(
    FATAL_ERROR
      "No pinned Dawn archive for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}"
  )
endif()
