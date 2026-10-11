# Package only scripted files using the same manifest as the extension's SDK build.
# Binary Slicer downloads contain no developer SDK. This is not a native build.
cmake_minimum_required(VERSION 3.28...3.31)
project(VesselRaySegmentationScriptedPackage LANGUAGES NONE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT SLICER_VERSION MATCHES "^[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "SLICER_VERSION must be the runtime's major.minor version")
endif()
get_filename_component(EXTENSION_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(module_source "${EXTENSION_SOURCE_DIR}/VesselRaySegmentationDemo")
include("${module_source}/ModuleFiles.cmake")
set(module_destination "lib/Slicer-${SLICER_VERSION}/qt-scripted-modules")
foreach(file IN LISTS MODULE_PYTHON_SCRIPTS MODULE_PYTHON_RESOURCES)
  configure_file("${module_source}/${file}" "staged/${module_destination}/${file}" COPYONLY)
endforeach()
configure_file("${EXTENSION_SOURCE_DIR}/LICENSE.txt"
  "staged/share/Slicer-${SLICER_VERSION}/VesselRaySegmentation/LICENSE.txt" COPYONLY)
add_custom_target(check_python ALL
  COMMAND "${Python3_EXECUTABLE}" -m compileall -q "${CMAKE_CURRENT_BINARY_DIR}/staged"
  VERBATIM)
# Bytecode is checked, but omitted so artifacts work with each runtime's Python.
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/staged/" DESTINATION .
  PATTERN "__pycache__" EXCLUDE)
set(CPACK_GENERATOR ZIP)
set(CPACK_PACKAGE_VERSION "0.0.0")
set(CPACK_PACKAGE_FILE_NAME "VesselRaySegmentation-scripted-Slicer-${SLICER_VERSION}")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
include(CPack)
