# Vessel inference

`vessel-inference` supplies shared ARG and ModelO inference. The base wheel has
no dependencies. The `numerical` extra selects only NumPy and SciPy and is tested
on Python 3.12 and 3.14. Optional extras select `arg`, `model-o`, `schemas`,
`registry`, `label` or `dawn`. Native inference and the MONAI Label environment
use Python 3.12 and ONNX Runtime 1.30.0.

ARG uses an exported ONNX graph, `plans.json` and `dataset.json`. It does not
install Torch, nnU-Net or training code. ModelO consumes a CT and ordered ZYX
polylines, generates ray inputs and executes through matched native Dawn assets.
Models and patient data are supplied separately.

ModelO uses exactly two ordered ray-feature channels, `hu_normalized` and
`position`, within its seven input tensors. Exports must declare that ordered
list as JSON in ONNX `feature_channels` metadata. The runtime validates names,
shapes and dtypes before session creation. Rays stop at their first patch or
volume exit, matching training. Output logits are checked for finite values,
thresholded strictly above zero and combined by boolean union. No final sigmoid,
probability accumulator, smoothing or morphological closing is required.

The shared numerical `build_patch_plans` keeps center placement, origin gathering
and output crop size separate. Numeric `overlap` sets the nominal voxel-arclength
stride `patch_size * (1 - overlap)`. `context_margin_voxels` reserves an inner
core for sparse center coverage, while `gather_margin_voxels` widens the same-branch
origin gather without expanding the output crop. Both default to zero. `overlap=None`
keeps one patch per origin and ignores the context margin; an explicit gather
margin also applies to that dense schedule. Core coverage may add existing centers
beyond the nominal stride, and fails if sampled origins cannot cover an endpoint.
These library controls do not change the hosted request or deployed model contract.

Numerical modules import independently of native execution and server packages.
`arg_inference.plans` is standard-library-only; importing ARG no longer imports
its execution dependencies. ModelO still needs ONNX for graph preparation, so its
native execution extra retains that dependency. The Label service retains its
upstream runtime requirements; it is not part of the desktop bundle.

## Build and test

From the repository root, with uv 0.12.23:

```sh
uv build deployment/inference --out-dir build/backend/dist
uv run --project deployment/inference --frozen --group test pytest deployment/inference/tests
```

The default `uv build` builds the wheel from the source distribution. Package
acceptance also installs the wheel into a fresh environment and checks its WGSL
resources. This catches missing files that source-tree tests cannot detect.

## Native assets

The Python wheel contains no native libraries. Build the separate matched asset
set using CMake 3.28+, Ninja and a C++20 compiler:

```sh
uv sync --project deployment/inference --frozen --group native --group test
cmake -S deployment/inference/native -B build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/deployment/inference/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/native/assets"
cmake --build build/native --parallel 2
cmake --install build/native
python tools/packaging/bundle_native.py --build build/native --output build/backend/native
```

`native/pins.cmake` fixes the Dawn and ORT source identities. The build verifies
Dawn headers against the provider's generated headers and applies the required
ConvTranspose3D patches. The bundle contains the host bridge, provider, generated
`webgpu.py`, upstream notices, platform identity and SHA-256 inventory. Distribute
these files together. ModelO's `--assets` option accepts the extracted
bundle directory. Do not mix this provider with another ORT version.

The build selects Metal on macOS, Vulkan on Linux and D3D12 on Windows. Linux
builds Dawn from source on both architectures. macOS and Windows use pinned
upstream archives. Configured targets are not a claim of hardware acceptance. Public CI
currently compiles Linux x86-64 and checks the bundle on Mesa software Vulkan.
Physical GPU regression tests require `DAWN_ORT_NATIVE_ASSETS` and explicitly
reject software and Null adapters.

The desktop packager combines the native asset ZIP, shared Python wheel, a private
CPython installation and locked `arg`/`model-o` dependencies. It does not install
packages into Slicer's interpreter. The host acceptance launches this packaged
worker and verifies dependency origins after relocation.

SlicerWeb uses the same numerical Python code in Pyodide. A separate
[WASM build](wasm/README.md) compiles the patched ORT C++ WebGPU provider and its
matching JavaScript bindings. The browser supplies WebGPU; desktop Dawn libraries
are not loaded there. Slang supplies the ray shader; WGSL and reflection are
generated package resources. See the canonical [build and validation guide](../../docs/build-and-test.md)
for packaging commands and proof boundaries.
