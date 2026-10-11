# Vessel inference

`vessel-inference` supplies shared ARG and ModelO inference. The base wheel has
no dependencies. Optional extras select `arg`, `model-o`, `schemas`, `registry`,
`label` or `dawn`. Python 3.12 and ONNX Runtime 1.30.0 are required for execution.

ARG uses an exported ONNX graph, `plans.json` and `dataset.json`. It does not
install Torch, nnU-Net or training code. ModelO consumes a CT and ordered ZYX
polylines, generates ray inputs and executes through matched native Dawn assets.
Models and patient data are supplied separately.

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
python ci/bundle_native.py --build build/native --output build/backend/native
```

`native/pins.cmake` fixes the Dawn and ORT source identities. The build verifies
Dawn headers against the provider's generated headers and applies the required
ConvTranspose3D patches. The bundle contains the host bridge, provider, generated
`webgpu.py`, upstream notices, platform identity and SHA-256 inventory. Distribute
these files together. ModelO's `--assets` option accepts the extracted
bundle directory. Do not mix this provider with another ORT version.

The build selects Metal on macOS, Vulkan on Linux and D3D12 on Windows. Linux
ARM64 builds Dawn from source; the other configured targets use pinned upstream
archives. Configured targets are not a claim of hardware acceptance. Public CI
currently compiles Linux x86-64 and checks the bundle on Mesa software Vulkan.
Physical GPU regression tests require `DAWN_ORT_NATIVE_ASSETS` and explicitly
reject software and Null adapters.

Native assets belong to desktop/server installations. They cannot execute in
SlicerWeb's WebAssembly Python. Browser clients use a separately hosted service;
that client integration is a later migration slice.
