# Optional compiled runtime payloads, shared by SDK and scripted CI packaging.
if(VRS_INFERENCE_WHEEL)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_inference_staging "${CMAKE_CURRENT_BINARY_DIR}/inference-package")
  file(REMOVE_RECURSE "${_inference_staging}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m zipfile -e "${VRS_INFERENCE_WHEEL}"
            "${_inference_staging}" COMMAND_ERROR_IS_FATAL ANY)
  install(
    DIRECTORY "${_inference_staging}/"
    DESTINATION "${module_destination}"
    PATTERN "__pycache__" EXCLUDE)
endif()
foreach(_runtime IN ITEMS DESKTOP WASM)
  if(VRS_${_runtime}_RUNTIME_DIR)
    if(NOT EXISTS "${VRS_${_runtime}_RUNTIME_DIR}/manifest.json")
      message(FATAL_ERROR "${_runtime} runtime requires its manifest.json")
    endif()
    string(TOLOWER "${_runtime}" _runtime_name)
    install(
      DIRECTORY "${VRS_${_runtime}_RUNTIME_DIR}/"
      DESTINATION
        "share/Slicer-${SLICER_VERSION}/VesselRaySegmentation/${_runtime_name}-runtime"
      USE_SOURCE_PERMISSIONS)
  endif()
endforeach()
