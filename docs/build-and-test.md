# CI build and validation

## Scripted modules and native modules

Slicer loads Python scripted modules directly. Building their delivery package
copies their declared scripts and resources to Slicer's module layout. The
application's own Python interpreter executes them. A C++ module must first be
compiled and linked against a matching Slicer developer build tree; the downloaded
desktop application does not contain that SDK.

The root `CMakeLists.txt` and module `CMakeLists.txt` follow Slicer's normal
extension build. With a matching developer build tree:

```sh
cmake -S . -B build/sdk -DSlicer_DIR=/path/to/Slicer-build -DBUILD_TESTING=ON
cmake --build build/sdk
ctest --test-dir build/sdk --output-on-failure
cmake --build build/sdk --target package
```

The free-runner jobs use a separate, scripted-only CMake/CPack project in `tools/packaging/`.
Both builds read `VesselRaySegmentation/ModuleFiles.cmake`, so file selection
does not drift between their packaging rules. The CI project compiles Python
bytecode as a syntax check, packages the scripts, resources and license, then
extracts the archive for application tests. Bytecode is omitted from the archive
because the target applications can have different Python versions.

This is a real scripted archive build, but it does not exercise the root
`find_package(Slicer)` configuration, Slicer CPack metadata or native compilation.
The archives are CI artifacts, not catalog-ready extension releases. Slicer's
extension build infrastructure must validate the standard SDK build and produce
the platform packages when the extension is submitted. The retired VPT C++ application is not included in this build.

## What the jobs prove

- Desktop stable and nightly use current official Linux application packages,
  not mocked Slicer imports. They import the extracted module, construct its
  five-stage widget, and exercise real MRML volumes, segmentations and curves.
- SlicerWeb uses the current `runtime-latest` wheels from
  `lassoan/slicerweb-app`. The runtime's `build-info.json` selects the matching
  `lassoan/SlicerWeb` source commit. Its wheel builder packages the CMake install
  tree. Chrome installs that wheel and runs the same MRML/module checks.
- Backend application jobs also start a temporary native MONAI Label endpoint.
  The actual widget discovers models, uploads one CT, runs input-dependent ARG,
  filters and trims a separate segmentation, submits a manual curve to ModelO,
  edits that curve and verifies the changed mask. They close a scene during a
  pending response and reject stale result imports. Browser requests cross the
  real fetch/CORS boundary. Endpoint dependencies live in a separate test
  environment and are not installed in Slicer or Pyodide.
- Automatic VMTK extraction is checked when ExtractCenterline is installed,
  including endpoint coverage. Otherwise the test verifies an explicit unavailable
  state; manual/imported editable Markups remain supported. SlicerWeb currently
  does not bundle VMTK.
- Local infrastructure tests use real temporary Git repositories to check
  publication rules, and CMake/CPack to check archive contents and source bytes.

Tests must complete with a successful process status. Exceptions, startup errors
and timeouts fail the job. Each target has an independent job; a failed nightly
does not prevent stable or browser results from being collected.

## Free runner setup

GitLab gates publication with the fast production checks and the acceptance jobs
in `.gitlab/acceptance.yml`. Acceptance runs for production/CI changes in branch
and merge-request pipelines, and always on default-branch and tag pipelines.
Cheap checks must pass before the native, WASM and container builds start.
The native and WASM producers then run independently. Their artifacts feed two
desktop matrix jobs and one browser job through explicit `needs` dependencies.
Those three jobs cover the scripted application checks as well as compiled
inference, so GitLab does not launch another copy of the basic application matrix.
All acceptance jobs are required by `publish:github`.

The jobs use Ubuntu 24.04 on `linux, large, amd64` runners, with three
compiler workers and a two-hour build limit. GitHub retains two compiler workers.
Container acceptance uses the
runner's existing Unix-socket Docker-in-Docker service. Its temporary bind mounts
remain under the shared project directory; `VRS_CONTAINER_HOST=docker` selects
the service network address without changing installed-wheel HTTP acceptance.
Both platforms invoke the same scripts under `tools/ci/`.
The BuildKit builder shares the job-local DinD network so private registry DNS
works on both executors. Cache export failures are nonfatal; image builds and
HTTP acceptance remain required. The optional `acceptance:packages:manual` job
repeats the package gate with a selected `CONTAINER_RUNNER_TAG` (default `large`)
when checking a different executor. It can start independently of the automatic
pipeline's prerequisites. Set the runner-tag variable when creating a web/API
pipeline; variables supplied when playing a manual job do not reselect its runner.

GitLab caches uv dependencies by lockfile and job role, and compiler results by
platform and pinned runtime/toolchain. Native and WASM compiler caches are
separate and bounded to 2 GiB each. Emscripten system libraries and npm downloads
are cached alongside the corresponding build. No virtual environment, application
download, CMake build tree or test result is restored from cache. Builds and tests
still run after every restore. GitLab retains its protected/unprotected cache
separation, with pipeline source and ref-protection state also present in keys.
Python caches are pruned with `uv cache prune --ci` before upload.
The cache and environment share the job filesystem so uv can link dependencies
instead of copying them. GitLab uses Fastzip's fastest archive mode and reports
transfer progress. Short acceptance jobs allow twenty minutes, including
Kubernetes startup and cache restoration.

BuildKit exports multi-stage container layers to the project's existing registry,
using branch-specific `ci-cache-*` tags and the default branch as a read fallback.
The extension project's daily registry cleanup removes matching tags older than
seven days, keeps the newest tag per image and preserves `ci-cache-main`.
Regular image tags are outside that cleanup policy.
Authentication uses the native CI registry credentials. The private/public trust
bundle is passed to the isolated BuildKit builder when the runner supplies one.
Final acceptance images stay in the job-local daemon. Compiled native/WASM
bundles are pipeline artifacts, retained for one day; reports remain for one week.

Heavy GitHub workflows run on public pull requests, version tags, manual dispatch
and once each Sunday. Extension CI starts at 04:27 UTC and Backend CI at 04:57 UTC.
Main publication pushes run the cheap Python quality workflow. GitHub restores
dependency caches and compiler caches bounded to 1 GiB per producer; it does not
need access to the private GitLab runners or registry.

The public repository uses `ubuntu-24.04` standard runners, read-only workflow
permissions and no repository secrets. GitHub provides standard hosted runner
minutes free for public repositories. Storage has a separate allowance, so retain
extension artifacts and reports for seven days. The private desktop runtime is
retained for one day and passed between jobs once, rather than uploaded again in
each application archive. These jobs do not
upload Slicer applications, the web application bundle or upstream runtime wheels
as artifacts. No larger runners, private GitLab runner access or hosted GPU is needed
for these tests. Chrome and native Vulkan can use explicitly reported software
GPU adapters.

The GitLab publication job pushes ordinary commits using the repository deploy
key, which triggers GitHub's Python quality workflow for matching paths. Direct pull requests can also be tested
on GitHub. Accepted source changes still belong in the authoritative GitLab
repository. GitHub CI has no permission to publish source changes or releases.

Publication is one-way from GitLab to GitHub. Make routine changes in GitLab,
including changes to `.github/workflows/`; publication carries those workflow
files to GitHub with the rest of the source. Extension CI also runs weekly on
Sunday at 04:27 UTC to check upstream stable, nightly and browser runtimes.

For contributions made on GitHub, fetch the contributor's branch and merge its
commits into a branch based on GitLab `main`. Open a GitLab merge request and run
the same checks before merging. If changes have already reached GitHub `main`,
merge that public history into GitLab so the next publication can fast-forward.
There is no automatic reverse synchronization. Preserve public commits rather
than force-pushing over them. Update the private parent's gitlink separately
after selecting the extension revision it should use.

## Run locally

### Python quality

`Python quality` runs required Pyrefly type checks separately from Ruff linting
and formatting. The root virtual development project pins Pyrefly 0.55.0 and
Ruff 0.15.6; it is not an installable extension or runtime dependency.

```sh
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
UV_PROJECT_ENVIRONMENT="$PWD/deployment/inference/.venv-typecheck" \
  uv sync --project deployment/inference --python 3.12 --frozen --all-extras --group test
UV_PROJECT_ENVIRONMENT="$PWD/deployment/monai/.venv-typecheck" \
  uv sync --project deployment/monai --python 3.12 --frozen --all-extras --group dev
uv run --frozen pyrefly check --config pyproject.toml
uv run --frozen pyrefly check --config deployment/inference/pyproject.toml
uv run --frozen pyrefly check --config deployment/monai/pyproject.toml
```

The application configuration covers the Slicer module, packaging/publication
tools and infrastructure/integration tests. It explicitly permits missing host
modules supplied by Slicer, SlicerVMTK and Pyodide; real application tests still
validate those APIs. Backend checks keep their package-specific environments so
native and MONAI dependency profiles are not mixed. No type-error categories are
disabled to pass these gates.

Infrastructure tests:

```sh
python3 -m unittest discover -s tests/infrastructure -v
```

Scripted archive:

```sh
python3 tools/packaging/build_scripted.py --slicer-version 5.13 --output build/package
```

For a downloaded desktop Slicer, replace the version and launcher path below with
those of that application. Linux needs its normal GUI libraries and Xvfb:

```sh
export PACKAGE_ROOT="$PWD/build/package/unpacked"
export TEST_REPORT="$PWD/build/package/desktop.json"
xvfb-run -a /path/to/Slicer --no-splash --no-main-window --testing \
  --additional-module-paths "$PACKAGE_ROOT/lib/Slicer-5.13/qt-scripted-modules" \
  --python-script "$PWD/tests/integration/slicer/desktop.py"
```

Browser tests need Python 3.12+, CMake, Make, Git, Node 22+ and Chrome:

```sh
python3 tools/runtimes/prepare_slicerweb.py --output build/web-runtime
# Use slicer_version printed above; the published runtime currently supplies it.
python3 tools/packaging/build_scripted.py --slicer-version 5.13 --output build/web-package
python3 tools/packaging/build_slicerweb.py --source build/web-runtime/SlicerWeb \
  --slicer-version 5.13 --build build/web-package
export SLICERWEB_WHEELS="$PWD/build/web-runtime/wheels"
export SLICERWEB_EXTENSIONS="$PWD/build/web-package/wheels"
(cd build/web-runtime/SlicerWeb/web && npm ci && npm run build:app)
node tests/integration/slicer/slicerweb.mjs build/web-runtime/SlicerWeb build/web-package
```

## Application workflow acceptance

The shipped module implements endpoint setup, ARG inference, connected-component
cleanup and trimming, editable centerlines, and ModelO inference. Cleanup uses
component count or minimum physical volume and preserves the raw ARG segmentation.
Trimming opens the host's Segment Editor on the selected cleanup segment.

Pure geometry/transport tests live in `VesselRaySegmentation/Testing/Python`.
Run them with the installed Label test environment to include real HTTP tests:

```sh
uv sync --project deployment/monai --frozen --group dev
VRS_TEST_ENDPOINT_PYTHON="$PWD/deployment/monai/.venv/bin/python" \
  deployment/monai/.venv/bin/python -m pytest -q VesselRaySegmentation/Testing/Python
```

Set the same `VRS_TEST_ENDPOINT_PYTHON` for the desktop command above. For browser
acceptance, add `--endpoint-python "$PWD/deployment/monai/.venv/bin/python"` to
`slicerweb.mjs`. Reports distinguish MRML tests, automatic VMTK availability,
synthetic endpoint workflow and compiled inference runtime acceptance.

These synthetic checks prove geometry, transport and application integration.
Real model weights and inference environments require separate provisioning;
patient-level segmentation quality and target hardware acceptance remain separate
release gates. Scripted packaging does not establish a catalog-ready SDK build.

## Upstream references

- [Slicer extension build and testing requirements](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
- [Slicer Ubuntu runtime prerequisites](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#ubuntu-24-04-noble-numbat)
- [Slicer extension and scripted-module templates](https://github.com/Slicer/Slicer/tree/main/Utilities/Templates)
- [SlicerWeb extension packaging](https://github.com/lassoan/SlicerWeb/blob/main/docs/extensions.md)
- [SlicerWeb runtime publication](https://github.com/lassoan/SlicerWeb#building)
- [GitHub Actions billing and public repositories](https://docs.github.com/en/billing/concepts/product-billing/github-actions)

## Backend validation

`backend-ci.yml` runs independently from the stable/nightly/SlicerWeb application checks.
It builds wheel and source distributions for both production packages using uv
0.12.23. A fresh dependency environment comes from the existing lockfile; built
wheels replace source-built packages before acceptance. `python -I` and package
origin assertions reject accidental source imports. Tests start the installed
`asgi:app`, use a temporary MLflow registry and execute an input-dependent ONNX
fixture in both direct and process modes through real HTTP. Container acceptance
also checks that the ARG worker has no Torch or nnU-Net installation.

The native job compiles Linux x86-64, bundles the CMake install tree with upstream
licenses and a platform/hash manifest, and tests shared-buffer execution after ZIP
extraction. Software Vulkan validates native integration on a free hosted runner.
Hardware acceptance still requires the selected real model and actual target GPU.
Current macOS local checks use the previously built matching asset set; they do
not claim a fresh macOS native build from this workflow. Windows and Linux ARM64
native builds are configured in CMake but are not part of this first CI slice.

The desktop job also builds an isolated Python 3.12.15 environment with locked
`arg` and `model-o` extras. The bundle contains no Torch, nnU-Net, MONAI Label,
MLflow or Kubernetes installation. ONNX remains necessary for ModelO graph
preparation; SimpleITK remains necessary for native image I/O. The ORT core is
the upstream 1.30.0 Python wheel. The source build compiles the patched WebGPU
provider and Dawn bridge, not a replacement core wheel.

The WASM job compiles that patched C++ WebGPU provider using Emscripten 4.0.23,
SIMD and Asyncify. It builds matching JavaScript bindings from the same checkout,
bundles upstream notices and records source, patch, toolchain and file hashes.
It uses the exact canonical Dawn commit with ORT's required Dawn patches. This
build is independent of SlicerWeb's prebuilt application runtime.

Both runtime jobs package their payload through `RuntimeFiles.cmake`, shared
with the standard Slicer SDK build. Desktop stable/nightly tests launch the
extracted private Python worker. SlicerWeb tests install the extension wheel in
both main and job-worker Pyodide contexts, verify the installed payload hashes,
and invoke the compiled provider from a Python job. Pyodide supplies NumPy/SciPy;
it does not install native ORT or desktop dependencies.

Runtime acceptance executes shared Python ModelO preparation and four synthetic
ConvTranspose3D models: both ARG kernel forms in FP32 and FP16. WGSL produces
changed input buffers, ORT consumes them and writes caller-owned GPU outputs,
and WGSL consumes those outputs. Each case runs three iterations across two
session lifecycles with CPU fallback disabled. The caller submits the chain
without input uploads or intermediate completion waits and performs one final
readback. Those counters observe caller operations; they do not measure internal
ORT/provider synchronization. FP16 is
reported as skipped if the device lacks `shader-f16`; use `--require-f16` for
hardware acceptance. This proves the runtime boundary, not full ModelO/ARG model
parity, clinical accuracy, the complete workflow or a catalog-ready SDK release.

Build a desktop package after compiling and bundling native assets:

```sh
uv build deployment/inference --out-dir build/backend/dist
python tools/packaging/bundle_desktop.py --native-archive build/backend/native/dawn-ort-linux-x86_64.zip \
  --wheel build/backend/dist/vessel_inference-0.1.0-py3-none-any.whl --output build/desktop-runtime
python tools/packaging/build_scripted.py --slicer-version 5.10 \
  --inference-wheel build/backend/dist/vessel_inference-0.1.0-py3-none-any.whl \
  --desktop-runtime build/desktop-runtime/desktop-linux-x86_64 --output build/desktop-package
build/desktop-runtime/desktop-linux-x86_64/python/bin/python3 tests/integration/runtime/fixtures.py --output build/runtime-fixtures
```

Use the desktop command above with `VRS_RUNTIME_FIXTURES="$PWD/build/runtime-fixtures"`.
Software-only hosts also need `VRS_RUNTIME_ALLOW_SOFTWARE=1` and
`ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER=1`. Select the native archive and bundle
directory for the current platform; use the downloaded Slicer's reported version.

For browser acceptance, build the [WASM component](../deployment/inference/wasm/README.md),
then add `--inference-wheel build/backend/dist/vessel_inference-0.1.0-py3-none-any.whl`
and `--wasm-runtime build/wasm/dist` to the scripted packaging command above.
After creating the SlicerWeb wheel and building its web app, invoke:

```sh
node tests/integration/slicer/slicerweb.mjs build/web-runtime/SlicerWeb build/web-package \
  --fixtures build/runtime-fixtures --allow-software
```

The separate shared-Python matrix accepts installed wheels on Python 3.12 and
3.14 with only NumPy/SciPy. Native/Label lockfiles remain scoped to Python 3.12
because the Label dependency set requires NumPy below 2, while newer Python
versions need newer numerical wheels. Generated WGSL/reflection resources are packaged from the typed Slang sources;
the compiler and SlangPy are not runtime dependencies.

Artifacts expire after seven days. Containers are built and tested locally on the
runner; this workflow does not publish registry images or create a release.

## Slang shader feasibility

Investigated on 2026-10-12. SlangPy 0.43.1 bundles Slang 2026.12, but its macOS
ARM64 wheel sets `SLANG_RHI_ENABLE_WGPU=0`; requesting a WebGPU device failed.
The inspected [upstream build configuration](https://github.com/shader-slang/slangpy/blob/ce68ef8d2079769d29c4234290ca326ff301ea8e/external/CMakeLists.txt)
also disables that backend explicitly, so a source build needs a configuration
change before testing WebGPU.

The [j8asic fork and upstream PR #787](https://github.com/shader-slang/slangpy/pull/787)
add browser WebGPU support to the C++ SGL library. At inspected revision
`d5fa8c5ccb88db86bab9012a28ad8cb4f6b292b7`, its Emscripten configuration explicitly
rejects `SGL_BUILD_PYTHON=ON`. It does not supply the SlangPy Python extension
needed by a SlicerWeb Pyodide worker. Its browser device tests are also separate
from its Node.js CI tests, which skip GPU devices.

[SlangPy PR #1090](https://github.com/shader-slang/slangpy/pull/1090) added native
buffer imports after the inspected wheel release. However, the inspected
[Slang-RHI WebGPU device implementation](https://github.com/shader-slang/slang-rhi/blob/242091b9bee857c46b1a70415afc35f2879853e4/src/wgpu/wgpu-device.cpp)
creates a new device, does not adopt `existingDeviceHandles`, and returns
`SLANG_E_NOT_IMPLEMENTED` for device-handle export. Its native backend loads a
separate Dawn library beside Slang-RHI. Runtime integration would therefore need
device interop changes, matching Dawn headers/library, and Python/Pyodide
packaging work. Importing a buffer handle cannot make a buffer usable on another
device.

The production migration uses typed `.slang` source compiled before packaging
into WGSL plus reflected resource bindings. Python keeps orchestration; the
existing native Dawn host and browser WebGPU host keep device and buffer
ownership. The compiler is a build dependency. SlangPy and Slang-RHI are not
required in the installed application. WebGPU still consumes generated WGSL;
the migration removes hand-written shader strings and derives binding metadata
from the compiler. [SlangWebGPU](https://github.com/eliemichel/SlangWebGPU) provides
an existing native/browser example of Slang compilation and reflected bindings.

A local feasibility probe compiled typed producer/consumer shaders with Slang
2026.12 from the isolated SlangPy wheel and substituted their generated WGSL into
the existing runtime fixtures. Native Dawn/Metal and a real SlicerWeb Python job
worker each passed eight FP32/FP16 cases across 24 changed-input inference
executions. CPU fallback was disabled and caller-owned GPU outputs were retained.
Native values exactly matched the original WGSL fixture results. Evidence lives
under ignored `build/shader-research/`, including `compiler.json`,
`native-metal.json` and `web-acceptance/results/slicerweb.json`.

### Production shader build and contract

The ModelO ray builder is authored in
`deployment/inference/src/model_o_inference/webgpu/patch_inputs.slang`. Slang 2026.12 generates the WGSL and reflection resources in
`webgpu/generated/`. Python does not rewrite shader source at runtime.

Regenerate after editing the shader or the shared constants:

```sh
python tools/packaging/compile_shaders.py
python tools/packaging/compile_shaders.py --check
```

The tool downloads an official compiler archive pinned by platform and SHA-256
into ignored `build/slang/`. Linux x86-64 and macOS ARM64 are supported build
hosts. `--compiler /path/to/slangc` uses an existing 2026.12 compiler. Generation
uses precise floating-point mode and records source/WGSL digests, compiler flags
and the shared Python constants in the reflection JSON. CI regenerates on Linux
and rejects stale checked resources. Ordinary source-distribution and wheel
builds consume those resources without downloading or installing a compiler.
SlangPy and Slang-RHI are not application dependencies.

The ray shader directly declares ten storage buffers, a 32-byte config record,
28-byte HU/Q16 ray records, byte-packed bool masks and two-word nonnegative
int64 indices. Three-scalar output records preserve ONNX's 12-byte `(N,3)` row
stride; plain float3 arrays would have a 16-byte stride under the
[WGSL layout rules](https://www.w3.org/TR/WGSL/#alignment-and-size). Each
120-sample ray owns complete packed mask words. Regeneration and resource loading
reject ray lengths not divisible by four.

| Owner | Responsibility |
| --- | --- |
| Slang sources | Sampling, two features and final GPU storage layout |
| Reflection | Resource names and binding indices, workgroup size, config field offsets |
| Python tensor contract | ONNX names, logical shapes, dtypes and byte lengths |
| Existing GPU hosts | Shared device/buffer ownership, dispatch, ORT binding and readback |

`shaders.py` loads and verifies resources. `shader_contract.py` retains semantic
ONNX types because reflection cannot infer that uint storage holds bool bytes or
int64 words. `dawn_inputs.py` binds buffers by reflected resource name and reuses
the request's CT buffer. Config marshalling supports the explicit contiguous
32-bit scalar record only; it rejects implicit layout changes rather than
attempting a general-purpose Slang serializer. Native and browser dispatches
use the reflected workgroup size.

### Production shader acceptance

`tests/integration/runtime/fixtures.py` generates the existing shared ORT cases
and `model_o/` fixtures from an independent NumPy sampling oracle.
The native runner executes the installed Python package. SlicerWeb's Python job
worker loads its installed shader resources and supplies them to the browser
adapter, which runs the same fixtures on the same device as ORT.

The four production cases exercise 65-ray workgroup tails, negative Q16 ties,
zero-valid rays, patch and volume exits, patch padding, constant and varying CT volumes,
zero patch normalization, all seven ONNX input layouts, and repeated buffer
reuse across changed configurations. A synthetic seven-input ONNX graph casts
and reduces each actual shader output on the shared device, verifying ORT's
interpretation of all seven layouts, including bool bytes and int64 words. CPU
fallback is disabled and scalar outputs remain in caller-owned GPU buffers.
The browser bridge sets `enableInt64: true`; the pinned ORT ESM patch forwards
this option to the C++ provider, matching native ModelO's `enableInt64="1"`.
Native acceptance also verifies that closing the builder preserves a borrowed
CT buffer. Shared Python thresholds logits strictly above zero, rejects NaN and
both infinities, and unions boolean patch masks. There is one output readback
per patch, with no sigmoid dispatch, probability buffer or invalid-flag buffer. These checks run inside the packaged stable/nightly desktop and
SlicerWeb backend workflow alongside the existing FP32/FP16 ORT tests.

Native `GpuSession` sets upstream `session.disable_prepacking=1` because optional
GPU weight prepacking produced intermittent bias-only ConvTranspose outputs on
Metal. Waiting after initialization alone did not prevent recurrence. Browser
acceptance uses the same policy. Initialization still completes uploads once
before exposing the session; repeated inference does not add caller completion
waits. Counters measure submissions separately from initialization. The provider's
internal cause and the performance cost of bypassing prepacking remain open.

The ray tensor contains only normalized HU and position. The position denominator
uses the number of samples before the first patch or volume exit, matching the
training adapter. Invalid HU/features are zero; invalid scatter indices are zero
and always masked. Gradient, prefix/suffix statistics, Fisher score and running-max
drop are removed, so there is no flat-profile CPU-parity exception.

`graph.py` requires exactly the seven named inputs, a two-channel feature tensor,
matching tensor dtypes and `mask_logits` output, plus ONNX metadata
`feature_channels='["hu_normalized", "position"]'`. Two channels alone cannot
identify the feature semantics. Legacy seven-channel and unnamed exports fail
explicitly instead of reaching a GPU shape error.

For private trained-model acceptance, generate fixtures with:

```sh
python tests/integration/runtime/fixtures.py --output build/runtime-fixtures \
  --model-o /path/to/selected/modelo.onnx
```

The generator validates and prepares the supplied graph, stages it under ignored
build output and computes CPU ONNX references. Native acceptance uses the actual
production submission, readback and binary-mask path. Browser acceptance runs the
same model with caller-owned GPU buffers and checks its masks in the SlicerWeb
Python worker. Neither path permits CPU fallback. Trained weights are not part
of the public repository or default synthetic CI fixtures.

Model logits use `atol=0.0005`, `rtol=0.003`, with exact binary-mask agreement.
These limits are distinct from the tighter shader-input comparisons. ModelO
preparation lowers large-spatial GroupNorm to centered, axiswise FP32 standard
ONNX operations while preserving learned affine parameters, epsilon and dynamic
ray counts. The original CPU ONNX reference itself differed from the untouched
PyTorch checkpoint; this lowering reduced the five-fixture maximum error from
0.0661 to 0.000145. Native and browser stress/foreground tests then passed the
unchanged logit limits with identical binary masks. The dense phantom contains
64 foreground voxels. Reports include foreground counts and distinguish shader
inputs, model scores and final masks. Matched patient cases remain necessary for
prediction-quality acceptance.

Q16 sampling splits signed coordinates into integer floor and fraction before
float conversion and computes normalized origins relative to the integer patch
corner. Native/browser regressions cover negative borders, signed half ties and
single-unit offsets at voxel coordinates 300 and 3072.

Strict zero thresholding implements the mathematical 0.5 probability boundary.
It deliberately keeps tiny positive logits that the previous FP32 sigmoid could
round to exactly 0.5. Boundary tests cover both signed zeros, positive/negative
small values, NaN and infinities. Boolean union matches maximum-logit decisions
and uses one byte per accumulated voxel instead of four.

Local cleanup evidence belongs under ignored `build/model-o-cleanup/`, including
checkpoint/export provenance, installed native reports and the SlicerWeb report.
Shader compilation, numerical comparison, resource packaging and installed-host
execution are separate gates. Synthetic fixtures establish these interfaces;
they do not establish patient-level prediction quality or clinical equivalence.
