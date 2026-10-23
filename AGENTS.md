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

The root `pyproject.toml` is a virtual development-tools project. Pyrefly checks
types; Ruff checks lint and formatting. Both are required by `Python quality` CI.
Use the package-specific typing environments documented in `docs/build-and-test.md`;
the root tool environment intentionally has no application runtime dependencies.

## Alerts

- ORT's automatic CPU thread count can follow the full node rather than a Slurm
  allocation's cpuset. On Grete this emitted affinity errors and stalled the
  CPU reference check. Keep explicit thread counts in CUDA host sessions and
  their CPU acceptance references; OpenMP environment settings alone do not
  configure ORT's thread pool.

- Private branch pipelines must run production checks before publication. The
  former publication-only job did not exercise inference, service or application
  code. Keep all fast test jobs in the publication dependency list; GitHub
  application/native acceptance still runs after protected-main publication.

- Linux ARM64 installs `numpymaxflow` from source. Service CI needs a C++
  compiler and Make even when the x86-64 job succeeds with a prebuilt wheel.

- GCC 14 enables C++ module dependency scanning for C++20 targets. The native
  bridge uses ordinary C API headers, so keep its `CXX_SCAN_FOR_MODULES` disabled.
  Otherwise Ninja's build-graph dry run requires a `CXX.dd` file that dry-run
  scanning never materializes. Keep the Linux ordering regression active.

- ModelO gathers origins around the continuous selected origin, but truncates that
  origin for the integer output crop. Half-open boundary tests must cover both
  coordinate conventions. The planner's two explicit margin keywords control
  independent geometry: context changes sparse center coverage, while gather
  changes the origins available to a fixed output crop. Keep both defaults zero
  for existing callers, and record any selected policy with its consumer.
  Its six-parameter public signature is intentional: the two margin controls stay
  explicit instead of hiding independent geometry in an options object solely
  to satisfy the usual five-parameter limit.

- Pyrefly's application search path must put `VesselRaySegmentation/` before
  the repository root. Reversing them confuses the module directory with its
  same-named Slicer entry point and reports missing internal package imports.

- Direct GitHub README edits during a GitLab outage caused publication to reject
  a non-fast-forward update. Fetch and preserve public-only commits by merging
  them into authoritative GitLab history before retrying. Never force-push the
  mirror or overwrite those edits.

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
  stitching in shared Python packages. Slang is now the shader source; generated
  WGSL/reflection are package resources verified by the pinned compiler in CI.

- SlangPy's WebGPU enum does not establish backend availability. The inspected
  0.43.1 macOS ARM64 wheel has `SLANG_RHI_ENABLE_WGPU=0`. The browser work in
  upstream PR #787 builds C++ SGL and explicitly rejects Python bindings under
  Emscripten. Slang-RHI's inspected WebGPU backend also lacks device-handle
  export and does not adopt `existingDeviceHandles`; native buffer import alone
  cannot share our ORT device. See the Slang feasibility section in the build
  guide before adding a runtime dependency or enabling a second Dawn library.

- ModelO now declares its final ten-buffer layout in Slang. Keep ONNX bool
  bytes and int64 words explicit, and use three-scalar records for 12-byte
  `(N,3)` rows. Do not replace them with padded float3 arrays or restore runtime
  WGSL rewriting. Reflection supplies bindings and config offsets, not ONNX
  semantic shapes/dtypes. See the build guide for regeneration and acceptance.

- Seven ONNX input tensors do not imply seven ray-feature channels. Production
  uses exactly `hu_normalized`, `position`, in that order. Some two-channel
  research checkpoints use different features, so ONNX `feature_channels`
  metadata is required as well as shape/dtype validation. Do not infer semantics
  from channel count. Legacy seven-channel exports are explicitly unsupported.
  The internal training exporter now applies the production graph preparation;
  desktop staging copies still have older contracts and are not release sources.

- Shape-valid ModelO exports can require different sampling semantics. Serving
  supports patch-truncated rays, legacy origin placement, trilinear origin HU
  and the native CT grid. The graph validator rejects conflicting policy metadata
  before session creation. Missing policy fields retain the historical shipped
  contract. Internal exports retain their actual policies after shared graph
  preparation; do not remove metadata to make an incompatible model deployable.

- The ray shader truncates at the first patch or volume exit, matching training.
  Position is normalized by that valid length; padded features and HU are zero.
  Do not restore full-volume features with only a scatter mask. Fisher statistics
  and the legacy flat-profile parity exception are absent from production.

- Binary output thresholds finite logits strictly above zero, then unions masks.
  Preserve NaN/infinity rejection when editing this path. Tiny positive logits
  can differ from the old FP32 sigmoid path, which rounded them to exactly 0.5;
  the new strict zero boundary is intentional and tested. There is no external
  sigmoid shader or floating-point probability accumulator.

- CPU ONNX is not automatically the trained-model reference. Original large
  spatial GroupNorm contributed most of the observed CPU/native discrepancy.
  Production graph preparation now lowers GroupNorm to centered, axiswise FP32
  standard operations, preserving affine state and epsilon. The same raw
  HU/position checkpoint then matches its untouched PyTorch forward within
  0.000145 across five fixtures. Native and browser foreground fixtures pass
  unchanged tolerances and identical masks. Keep native checkpoint, model-score,
  near-zero decision and final-mask checks separate; this is not patient validation.

- Keep Q16 floor, fractions and ties in integer arithmetic. The Slang shader
  uses arithmetic shifts and masks before interpolation and subtracts the
  integer patch corner before origin normalization. Whole-coordinate float32
  conversion loses fractional bits above voxel 256. Regress signed borders,
  half ties and one-Q16-unit offsets on both native and browser runtimes.

- SlicerWeb currently has no Python sockets, bundled VMTK or usable native
  QLineEdit password echo mode. Hosted requests use asynchronous browser fetch,
  with CORS configured by the endpoint. The browser adapter must mask the bearer
  token's DOM input; desktop uses native password echo mode. Never persist bearer
  tokens in settings, MRML scenes or logs. Automatic centerlines require the
  available ExtractCenterline module; manual/imported Markups are supported.

- ORT resolves relative provider-library paths beneath its own C API directory.
  Native acceptance canonicalizes asset paths before registering the library.

- The pinned ORT browser API omitted forwarding `enableInt64` to its C++ WebGPU
  provider. The WASM ESM patch adds the typed boolean and forwards it explicitly.
  Keep the seven-input ONNX acceptance with CPU fallback disabled; the earlier
  floating-point fixtures could not detect this missing ModelO requirement.

- The pinned provider's optional weight prepacking can produce bias-only native
  ConvTranspose outputs despite correct caller inputs. An initialization wait
  did not prevent recurrence. `GpuSession` now sets upstream
  `session.disable_prepacking=1`, retaining original weights for its direct
  kernels; browser acceptance uses the same policy. Initialization still
  completes uploads before caller work, and per-inference submissions remain
  asynchronous. Do not enable prepacking without repeated fresh-process
  numerical acceptance. The underlying provider cause remains unassigned.

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

- Slicer launches can export PYTHONHOME and library paths for their embedded
  Python. Sanitize that inherited environment when launching an independent
  Python service; otherwise its own standard library can fail before startup.

- Browser fetch timeouts must cover response-body reading as well as receipt of
  headers. Abort pending reads, and never launch inference after session disposal
  during discovery/upload. Closing a MRML scene discards pending imports and
  releases the client snapshot; it does not prove remote inference cancellation.
  Before import, verify that the live CT still matches the captured identity,
  grid, voxel values and transform policy; another module can edit it while
  the workflow panel is disabled.
