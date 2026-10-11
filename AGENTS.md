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
- Alpine's CMake package does not include Make. The GitLab test image installs
  both; otherwise even a scripted-only CMake project fails before packaging.
- The hosted Ubuntu image needs Slicer's documented Qt/XCB runtime prerequisites.
  A partial hand-picked list omitted `libxcb-shape.so.0` and prevented both desktop
  applications from starting. Keep the documented Ubuntu 24.04 dependency set.

- The original MONAI wheel omitted `asgi.py` even though the documented startup
  command imports it. Keep installed-wheel acceptance outside the checkout;
  source-directory imports and Docker COPY instructions can hide omitted files.

## Backend migration

`deployment/inference` and `deployment/monai` are the public production backend.
Keep research, trained weights, private runbooks and cluster settings internal.
`deployment/migration-source.json` records the initial working-tree snapshot; it
includes five existing source edits without changing their internal originals.
The internal checkout remains a staging copy during client migration. Make new
backend changes here; reconcile internal callers explicitly before deleting or
replacing their existing files. Do not maintain two independently evolving copies.

Run the backend workflow as well as the extension workflow. A Python wheel does
not include native Dawn assets; native bundle checks and real hardware acceptance
are distinct. Keep CPU container inference and installed-wheel HTTP checks real.

- Unified patch files contain space-prefixed blank context lines. Git's ordinary
  trailing-whitespace check flags those lines when a patch is first added as a
  file. `.gitattributes` disables that check only for `*.patch`; do not strip
  patch context to satisfy formatting checks.

- The pinned upstream Linux x86-64 Dawn archive passed its download checksum but
  failed the procedure-table identity check during the first clean CI build.
  Linux now builds Dawn from the exact pinned source, as ARM64 already did. Keep
  the runtime identity gate strict; a filename and checksum do not prove ABI
  identity. Binding-generation errors report expected and actual identities.

- The first Docker acceptance executed inference successfully but left root-owned
  cache directories in the runner's temporary mount, so cleanup failed. CI now
  runs server and worker containers with the runner's UID/GID. Label's image
  creates its named `monai` account using matching build arguments and gives it
  a writable native app directory. Keep cleanup part of the acceptance gate.

- ORT's pinned WebGPU factory advertises only hardware GPU devices by default.
  Hosted native CI explicitly sets `ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER=1` for
  its Mesa check. Keep that opt-in confined to software acceptance and report
  the actual adapter; it does not establish physical GPU execution.
