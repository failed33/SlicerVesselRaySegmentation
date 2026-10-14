cmake_minimum_required(VERSION 3.28...3.31 FATAL_ERROR)
project(VesselRaySegmentation)

set(EXTENSION_HOMEPAGE
    "https://github.com/failed33/SlicerVesselRaySegmentation")
set(EXTENSION_CATEGORY "Vascular Modeling")
set(EXTENSION_CONTRIBUTORS "Moritz Kirschner")
set(EXTENSION_DESCRIPTION
    "ARG segmentation, editable vessel cleanup, centerlines and ModelO inference."
)
set(EXTENSION_ICONURL
    "${EXTENSION_HOMEPAGE}/raw/main/VesselRaySegmentation/Resources/Icons/VesselRaySegmentation.png"
)
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})
add_subdirectory(VesselRaySegmentation)
set(SLICER_VERSION "${Slicer_VERSION_MAJOR}.${Slicer_VERSION_MINOR}")
set(module_destination "lib/Slicer-${SLICER_VERSION}/qt-scripted-modules")
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/packaging/RuntimeFiles.cmake")
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
