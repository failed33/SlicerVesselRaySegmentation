include(${CMAKE_CURRENT_SOURCE_DIR}/ModuleFiles.cmake)
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
)
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
