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
Both builds read `VesselRaySegmentationDemo/ModuleFiles.cmake`, so file selection
does not drift between their packaging rules. The CI project compiles Python
bytecode as a syntax check, packages the scripts, resources and license, then
extracts the archive for application tests. Bytecode is omitted from the archive
because the target applications can have different Python versions.

This is a real scripted archive build, but it does not exercise the root
`find_package(Slicer)` configuration, Slicer CPack metadata or native compilation.
The archives are CI artifacts, not catalog-ready extension releases. Slicer's
extension build infrastructure must validate the standard SDK build and produce
the platform packages when the extension is submitted. Retained C++ components
would also need their own SDK build before a release.

## What the jobs prove

- Desktop stable and nightly use current official Linux application packages,
  not mocked Slicer imports. They load the unpacked package through additional
  module paths, assert the module was imported from it, check a real MRML curve's
  control points and length, construct the module widget through Slicer's module
  manager and click its button to check the scene mutation.
- SlicerWeb uses the current `runtime-latest` wheels from
  `lassoan/slicerweb-app`. The runtime's `build-info.json` selects the exact
  matching `lassoan/SlicerWeb` source commit. Its own wheel builder packages the
  CMake install tree. Chrome loads the built web application, installs the wheel
  through SlicerWeb's runtime and runs the identical module tests.
- Local infrastructure tests use real temporary Git repositories to check
  publication rules, and CMake/CPack to check archive contents and source bytes.

Tests must complete with a successful process status. Exceptions, startup errors
and timeouts fail the job. Each target has an independent job; a failed nightly
does not prevent stable or browser results from being collected.

## Free runner setup

The public repository uses `ubuntu-24.04` standard runners, read-only workflow
permissions and no repository secrets. GitHub provides standard hosted runner
minutes free for public repositories. Storage has a separate allowance, so retain
extension artifacts and reports for seven days. The private desktop runtime is
retained for one day and passed between jobs once, rather than uploaded again in
each application archive. These jobs do not
upload Slicer applications, the web application bundle, upstream runtime wheels
or caches. No larger runners, private GitLab runner access or hosted GPU is needed
for these tests. Chrome and native Vulkan can use explicitly reported software
GPU adapters.

The GitLab publication job pushes ordinary commits using the repository deploy
key, which triggers GitHub's push workflow. Direct pull requests can also be tested
on GitHub. Accepted source changes still belong in the authoritative GitLab
repository. GitHub CI has no permission to publish source changes or releases.

## Run locally

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

Browser tests need Python 3.12+, CMake, Make, Git, GitHub CLI, Node 22+ and Chrome:

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

## Validate the vessel application after migration

Keep the same compatibility jobs, replacing the demo with the real module and its
tests. Add algorithm tests with deterministic synthetic masks and centerlines,
then an end-to-end fixture through ARG, disconnected-component removal, centerline
generation and editing, and ModelO. Define expected geometry and segmentation
metrics before accepting changes. Hosted inference also needs integration tests
for transport, cancellation, model selection and result import.

Real model weights and inference environments require separate provisioning.
This demo's software-rendered browser and desktop tests do not establish model
accuracy or native GPU/provider compatibility. Release checks must cover the
actual supported deployment and desktop platforms.

## Upstream references

- [Slicer extension build and testing requirements](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
- [Slicer Ubuntu runtime prerequisites](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#ubuntu-24-04-noble-numbat)
- [Slicer extension and scripted-module templates](https://github.com/Slicer/Slicer/tree/main/Utilities/Templates)
- [SlicerWeb extension packaging](https://github.com/lassoan/SlicerWeb/blob/main/docs/extensions.md)
- [SlicerWeb runtime publication](https://github.com/lassoan/SlicerWeb#building)
- [GitHub Actions billing and public repositories](https://docs.github.com/en/billing/concepts/product-billing/github-actions)

## Backend validation

`backend-ci.yml` runs independently from the stable/nightly/SlicerWeb demo checks.
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
versions need newer numerical wheels. Existing WGSL resources remain packaged;
the PySlang rewrite is deferred.

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

The recommended migration is typed `.slang` source compiled before packaging
into WGSL plus reflected resource bindings. Python keeps orchestration; the
existing native Dawn host and browser WebGPU host keep device and buffer
ownership. The compiler is a build dependency. SlangPy and Slang-RHI would not
be required in the installed application. WebGPU still consumes generated WGSL;
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

This probe validates the compilation and shared-buffer boundary. It does not
port the production ModelO ray shader, validate a SlangPy WebGPU runtime, rerun
stable/nightly desktop Slicer, or establish full-model parity. The production
port still needs tests of fixed-point rounding, packed bool/int64 outputs,
volume boundaries, ray features and CT patch sampling against the numerical
reference, followed by installed-package acceptance on all three hosts.

### Concrete ModelO cleanup

The production source currently declares twelve buffers. `packed_shader()` and
`_compact_bindings()` then rewrite its text into ten bindings, combine config
and ray geometry buffers, change index storage to two words, and pack four mask
bytes into each word. `DawnPatchInputs` separately overrides output dtypes by
numeric binding. The declared source, host contract and executed program thus
describe different layouts. The Slang port should declare the final layout
directly and remove both source rewriting functions and the dtype overrides.

| Owner | Responsibility after the port |
| --- | --- |
| Slang source | Ray sampling, features, sigmoid, final storage types and resource declarations |
| Compiler reflection | Resource names, binding indices, entry point, thread group size and config field offsets |
| Python tensor contract | ONNX input names, logical shapes, dtypes and required byte lengths |
| Existing GPU hosts | Device and buffer lifetime, dispatch, ORT binding and completed readback |

Reflection cannot infer that a `uint` mask buffer represents an ONNX bool tensor,
or that two `uint` words represent one nonnegative int64 index. Keep those
semantic declarations explicit in `shader_contract.py`, keyed by resource name.
Use reflection to assemble the bindings from a name-to-buffer mapping, rather
than duplicating numeric indices or relying on positional `zip` operations.
Validate expected names and access modes and reject unsupported groups/layouts.
Keep this adapter limited to the storage buffers used here; a general shader
binding framework is unnecessary.

A second local probe used the official Slang 2026.12 macOS ARM64 compiler CLI,
including its working `-reflection-json` option. It declared ten resources,
a 32-byte config record and a 28-byte ray geometry record. The config stores
volume dimensions, ray count, integer patch corner and `h0`; the existing
60-byte upload also carries spacing, oversampling and travel settings that the
executed shader does not read. Fixed-point scale and training-locked dimensions
belong to the compile-time constants, with a generated-values check against the
Python constants.

The probe passed two dispatches on real Dawn/Metal (Apple M4 Pro) and in a Chrome
155 WebGPU worker using Apple's non-fallback Metal adapter. All seven output
buffers matched the independent NumPy reference byte for byte. This includes
negative Q16 half-way rounding, packed bool bytes, two-word int64 storage,
28-byte input records and 12-byte `(N,3)` output rows. A three-scalar structure
preserves the latter layout; `StructuredBuffer<float3>` would instead have a
16-byte array stride under the [WGSL layout rules](https://www.w3.org/TR/WGSL/#alignment-and-size).
The current 120-sample ray length lets each ray own complete four-byte mask
words, avoiding cross-thread read/modify/write races. Preserve and assert that
invariant when packing masks.

Evidence is under ignored `build/shader-research/layout/`: `layout.slang`,
generated WGSL/reflection, `native.json` and `browser.json`. This probe uses
synthetic arithmetic to test layout and rounding. It does not compute the
production ray features or exercise this layout inside a SlicerWeb Python job.
The earlier SlicerWeb job probe above establishes the compiler/ORT shared-buffer
boundary separately. The raw CLI reflection includes member offsets but does
not expose every resource record stride explicitly; do not infer arbitrary
structure/array strides from the final member's end offset. Restrict supported
layouts and verify their exact byte representation with these GPU tests.

### Source and packaging arrangement

Keep the two shader sources and their generated resources together beneath the
existing `src/model_o_inference/webgpu/` directory:

```text
webgpu/
  patch_inputs.slang
  sigmoid.slang
  generated/
    patch_inputs.wgsl
    patch_inputs.json
    sigmoid.wgsl
    sigmoid.json
```

`shaders.py` becomes a resource loader. Keep the numerical tensor contract in
`shader_contract.py` and orchestration in `dawn_inputs.py`. Extract shared Slang
helpers only when both kernels actually need them. Keep the same native Dawn
and browser WebGPU device shared with ORT, including borrowed CT buffer reuse.

Use one `tools/packaging/compile_shaders.py` command invoking the official
`slangc` CLI, pinned by release and platform archive checksum. Compile on the
build host, including when the application target is WASM. Record compiler
version, flags, source digest and the training constants with the generated
metadata. Start with the verified 2026.12 compiler; upgrade it deliberately
through the same tests as a source change. Do not add ctypes compiler bindings,
SlangPy or Slang-RHI to the installed inference dependencies.

Check in generated WGSL/reflection as package resources and have a CI
`--check` mode regenerate into a temporary directory and reject differences.
Regeneration is mandatory in CI and when editing shaders, while ordinary
`uv build` and source-distribution wheel builds consume the checked resources.
No custom wheel build hook or
compiler download is needed for those consumers. The current Hatch source
inclusion already covers this resource directory; installed-wheel tests must
check the new files and their matching digests.

Implement in two steps: first move sigmoid and introduce compiler/resource
checks, then port the ray shader and delete the WGSL rewriting path. Keep the
old shader only as a temporary comparison fixture during that port. Require
CPU-reference checks for all seven production outputs, volume exits, patch
padding, interpolation and feature accumulation; retain the packed-layout and
changed-input/lifetime regressions. Preserve accumulation order during the port
and assess numerical changes from compiler optimizations separately. Then run
installed-wheel stable Slicer,
nightly Slicer and SlicerWeb acceptance with the shared ORT buffers. Extension
packaging, shader compilation, numerical parity and execution inside each host
are separate validation gates. The layout probe completes only the relevant
layout checks, not the production migration or full-model acceptance.
