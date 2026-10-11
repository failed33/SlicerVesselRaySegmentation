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
only small extension artifacts and reports for seven days. These jobs do not
upload Slicer applications, the web application bundle, upstream runtime wheels
or caches. No larger runners, private GitLab runner access or hosted GPU is needed
for this demo. Chrome uses software rendering.

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

Artifacts expire after seven days. Containers are built and tested locally on the
runner; this workflow does not publish registry images or create a release.
