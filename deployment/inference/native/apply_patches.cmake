# This checkout belongs to ExternalProject. Preserve any unrelated local edits.
execute_process(
  COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
  WORKING_DIRECTORY "${SOURCE_DIR}"
  OUTPUT_VARIABLE _source_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT _source_commit STREQUAL EXPECTED_COMMIT)
  message(
    FATAL_ERROR
      "ORT source cache has the wrong commit. Use a fresh native build directory."
  )
endif()

set(_patches "${RUNTIME_PATCH}" "${WGSL_PATCH}")
file(SHA256 "${RUNTIME_PATCH}" _runtime_sha256)
file(SHA256 "${WGSL_PATCH}" _wgsl_sha256)
string(SHA256 _patch_set
              "${EXPECTED_COMMIT};${_runtime_sha256};${_wgsl_sha256}")
if(EXISTS "${PATCH_STATE_FILE}")
  file(READ "${PATCH_STATE_FILE}" _applied_set)
  if(NOT _applied_set STREQUAL _patch_set)
    message(
      FATAL_ERROR
        "Native ORT patch set changed. Use a fresh native build directory.")
  endif()
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --check ${_patches}
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _can_apply
  ERROR_VARIABLE _apply_error)
if(_can_apply EQUAL 0)
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply ${_patches}
                  WORKING_DIRECTORY "${SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY)
else()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check ${_patches}
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _already_applied
    ERROR_QUIET)
  if(NOT _already_applied EQUAL 0)
    message(FATAL_ERROR "Cannot apply native ORT patches to cached source. "
                        "Use a fresh native build directory.\n${_apply_error}")
  endif()
endif()
file(WRITE "${PATCH_STATE_FILE}" "${_patch_set}")
