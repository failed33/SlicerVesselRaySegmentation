# WebGPU WASM runtime

Build the ONNX Runtime C++ WebGPU execution provider with the same source pins
and ConvTranspose3D/WGSL patches as the native runtime:

```sh
python tools/packaging/build_wasm.py --output build/wasm --jobs 2
```

Use Python 3.12+, Git, CMake 3.28+, Ninja and Node 24. The builder installs the
Emscripten version in `toolchain.json` using ORT's pinned emsdk submodule. Sources,
downloaded dependencies, compiler output, npm caches and the build log stay under
the selected ignored `build/` directory. `--source-cache /path/to/onnxruntime`
can seed a fresh checkout from an existing Git object store without copying its
working-tree changes. `--dawn-source-cache` provides the equivalent Dawn seed.
Keep using the same output directory to resume a build.
Changed source pins or patches require a fresh output directory.

The build uses upstream `tools/ci_build/build.py --build_wasm --use_webgpu`, SIMD
and Asyncify. Dawn comes from the exact `DAWN_COMMIT` in `native/pins.cmake`, with
ORT's six upstream Dawn patches applied before the FetchContent source override.
ORT's default Dawn release archive resolves to a different source commit. The
manifest records actual source and generated header hashes.
It does not enable JSEP or compile SlicerWeb. ORT's standard WebGPU
build generates WGSL using the native provider; existing shader assets remain
WGSL. The small `ort-webgpu-esm.patch` adds an ESM output to upstream's
`npm run build -- --bundle-mode=perf --webgpu-ep`. The complete production bundle
matrix requires additional CPU, JSEP and JSPI artifacts that this component does
not ship. The selective build returns after emitting the ESM, before upstream's
separate test bundle that requires test-runner-generated fixture metadata.
Generated-checkout CommonJS and TypeScript type roots keep enclosing application
configuration from changing the upstream build; the manifest records their hashes.

`build/wasm/dist` contains:

- `ort.webgpu.min.mjs`, built from that same patched ORT checkout;
- `ort-wasm-simd-threaded.asyncify.mjs` and `.wasm`, compiled locally;
- upstream licenses, third-party notices and `manifest.json` with exact source,
  toolchain, patch and file hashes.

The matching versioned ZIP lives alongside `dist`. The manifest's `bundle_id`
includes the ORT version, source commit, patches and builder identity. No
prebuilt npm runtime is substituted when compilation fails.

In a browser worker, import `ort.webgpu.min.mjs`, set `ort.env.wasm.wasmPaths` to
the bundle directory URL, and set `numThreads=1` and `proxy=false`. The WASM is
compiled with thread support, but this invocation does not create pthread
workers or require a shared-memory Pyodide build. Select only the `webgpu`
execution provider and set `extra.session.disable_cpu_ep_fallback="1"` in
acceptance tests. The browser supplies the WebGPU implementation.

Compilation and packaging do not establish inference correctness. The runtime
acceptance tests must load the emitted files in real SlicerWeb, execute a rank-five
ConvTranspose fixture and report the actual adapter and numerics. Free GitHub
runner tests may use a software WebGPU adapter; report that explicitly. Desktop
Slicer uses the separately compiled native runtime.
