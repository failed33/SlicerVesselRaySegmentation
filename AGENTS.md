# Extension repository

GitLab is authoritative. Commit and push changes there, then verify publication
and the corresponding GitHub Actions run. Every file and commit here must be
suitable for public release. Keep research, model weights and private deployment
settings outside this repository.

Keep the standard Slicer SDK build in the root CMake project. Free-runner scripted
packaging lives in `tools/packaging/CMakeLists.txt`; both use the module's
`ModuleFiles.cmake`.
Do not describe scripted-package checks as SDK/native build validation.

Run `python3 -m unittest discover -s tests/infrastructure -v` with Python 3.12+,
Git and CMake 3.28+. Module integration tests must also pass in real stable Slicer, nightly
Slicer and SlicerWeb. Do not replace those gates with mocked imports.

Keep build and publication commands under `tools/`, application/package acceptance
under `tests/integration/`, and infrastructure unit tests under
`tests/infrastructure/`. Detailed instructions belong in `docs/build-and-test.md`.
Put downloads, extracted packages, virtual environments and reports under ignored
`build/`. Do not add generated artifacts or one-off proof files to the root.

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

- SlicerWeb's `installWheel()` updates the main Python filesystem only. Its job
  worker installs extension URLs from the application's startup configuration.
  Runtime acceptance must configure the tested wheel before page startup, or
  the main module can pass while the worker lacks the packaged application code.

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

- The existing desktop/browser checks use different inference runtimes. They do
  not validate one WASM inference component on both hosts. Desktop Slicer's local
  QtWebEngine is based on Chromium 87 and cannot provide modern WebGPU. A shared
  WASM component needs a tested GPU host on desktop; a Pyodide side-module wheel
  cannot be imported by native CPython.

- The application remains Python. Compile ORT/Dawn for native desktop and ORT's
  C++ WebGPU provider to WASM for SlicerWeb; keep preprocessing, orchestration and
  stitching in shared Python packages. WGSL remains the shader source for this
  rollout. The planned PySlang migration is separate work.

- SlangPy's WebGPU enum does not establish backend availability. The inspected
  0.43.1 macOS ARM64 wheel has `SLANG_RHI_ENABLE_WGPU=0`. The browser work in
  upstream PR #787 builds C++ SGL and explicitly rejects Python bindings under
  Emscripten. Slang-RHI's inspected WebGPU backend also lacks device-handle
  export and does not adopt `existingDeviceHandles`; native buffer import alone
  cannot share our ORT device. See the Slang feasibility section in the build
  guide before adding a runtime dependency or enabling a second Dawn library.

- ModelO's checked WGSL is currently rewritten at runtime from twelve bindings
  into ten, with packed bool/int64 storage and separate dtype overrides. Do not
  treat its source declarations as the executed buffer contract. A Slang port
  must declare the final layout directly and preserve ONNX's packed bytes and
  12-byte `(N,3)` rows; plain `float3` storage arrays have a 16-byte stride.
  The concrete cleanup and GPU layout evidence are in the build guide.

- Python metadata and runtime dependency support are different boundaries.
  Shared numerical code supports Python 3.14 in Pyodide, but the native/Label
  developer environments remain Python 3.12. Label's numpymaxflow needs NumPy 1,
  while newer ONNX/ml-dtypes selections on Python 3.13+ need NumPy 2. Keep the
  numerical and server dependency selections separate instead of forcing one
  incompatible environment.

- ORT's pinned Dawn release tag resolves to a different commit than the native
  asset pin, despite sharing the release name. The WASM builder must override
  Dawn with the exact canonical source commit and record its actual identity.
  Do not infer a source commit from the release label.

- ZIP extraction does not restore executable permissions automatically. The
  scripted packager restores CPack's Unix mode bits so the bundled private
  Python remains executable after extraction. Managed Python's internal
  symlinks are materialized before packaging.

- ORT's CommonJS WASM post-build script can inherit `type: module` from an
  enclosing internal repository because its source root has no package.json.
  The isolated builder creates a local CommonJS package boundary in ignored
  build output. Keep that boundary explicit when building inside a parent repo.
  TypeScript also searches enclosing `node_modules/@types` by default; scope
  its automatic type roots to ORT's own installed dependencies so internal Bun
  globals cannot enter the build.

- Runtime acceptance counters do not observe all ORT/Dawn activity. A native
  mutation check inserted 24 raw GPU completion waits and still passed with
  zero Python-host waits. Browser counters wrap selected JavaScript queue calls
  and do not count `GPUBuffer.mapAsync` waits. Keep report fields and claims
  scoped to the observed caller operations; they do not prove that the provider
  performs no internal synchronization or transfers.
