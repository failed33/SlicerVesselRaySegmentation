# Extension repository

GitLab is authoritative. Commit and push changes there, then verify publication
and the corresponding GitHub Actions run. Every file and commit here must be
suitable for public release. Keep research, model weights and private deployment
settings outside this repository.

Keep the standard Slicer SDK build in the root CMake project. Free-runner scripted
packaging lives in `ci/CMakeLists.txt`; both use the module's `ModuleFiles.cmake`.
Do not describe scripted-package checks as SDK/native build validation.

Run `python3 -m unittest discover -s tests -v` with Python 3.12+, Git and CMake
3.28+. Module integration tests must also pass in real stable Slicer, nightly
Slicer and SlicerWeb. Do not replace those gates with mocked imports.

## Alerts

- macOS CPack's tar generator can add AppleDouble `._` metadata files even when the
  source manifest contains none. CI's portable scripted artifact uses ZIP. Keep
  the archive content assertion strict rather than ignoring unexpected files.
