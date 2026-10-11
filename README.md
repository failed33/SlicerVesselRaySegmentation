# SlicerVesselRaySegmentation

This repository contains the shared ARG/ModelO inference backend, MONAI Label
service, publication infrastructure and a scripted module demo for a planned
3D Slicer extension. The vessel workflow UI is still being migrated.

The planned workflow is ARG inference, disconnected-component removal, centerline
generation and manual editing, then ModelO inference.

The extension's CMake project is `VesselRaySegmentation`; the repository name has
Slicer's conventional `Slicer` prefix. Its current module is
`VesselRaySegmentationDemo`, following the Module/Widget/Logic/Test pattern. Load
that folder through Slicer's additional module paths and open **Vessel Ray
Segmentation Demo** to create a synthetic, editable three-point Markups curve.

## Repository layout

| Path | Contents |
| --- | --- |
| `VesselRaySegmentationDemo/` | Slicer module, resources and in-application tests |
| `deployment/inference/` | Installable ARG/ModelO runtime and pinned native build |
| `deployment/monai/` | MONAI Label service, worker images and service tests |
| `tools/packaging/` | Scripted archive, SlicerWeb wheel and native bundle builders |
| `tools/runtimes/` | Upstream application downloads and runtime preparation |
| `tools/publication/` | GitLab-to-GitHub publication |
| `tests/infrastructure/` | Packaging and publication unit tests |
| `tests/integration/` | Tests executed against built packages and real applications |
| `docs/` | Build, test and release instructions |
| `build/` | Ignored downloads, environments, build output and reports |

## Build and test

[Extension CI](https://github.com/failed33/SlicerVesselRaySegmentation/actions/workflows/extension-ci.yml)
runs on pushes to `main`, version tags, pull requests, manual dispatch and daily
at 04:27 UTC. It uses standard public GitHub-hosted Ubuntu runners:

| Target | Build | Tests |
| --- | --- | --- |
| Current stable Slicer | Scripted CPack archive | Module discovery, MRML geometry and actual Qt button action |
| Current nightly Slicer | Scripted CPack archive | The same tests in the nightly application |
| Published SlicerWeb runtime | Scripted package and upstream extension wheel | Install the wheel and run the same tests in headless Chrome |

Artifacts include the small packages, test reports and upstream runtime provenance.
They expire after seven days. Application downloads and runtime wheels are not
uploaded or cached. Slicer downloads are checked against the size and SHA-512 in
the official download metadata. SlicerWeb application sources are checked out at
the exact unmodified commit recorded by its published runtime wheels.

These jobs validate the Python packaging and real application integration. The
demo checks do not validate ARG/ModelO inference, GPU behavior, the complete vessel
workflow, Windows/macOS, or a native C++ build.

See [build and validation](docs/build-and-test.md) for commands, the distinction
between scripted packaging and an SDK build, and the remaining release validation.

## Backend packages and native assets

[Backend CI](https://github.com/failed33/SlicerVesselRaySegmentation/actions/workflows/backend-ci.yml)
builds the two Python wheels from source distributions, accepts them outside the
checkout, and repeats synthetic ARG inference in built CPU containers. A separate
Linux job compiles the pinned Dawn host and patched ONNX Runtime provider, creates
a versioned ZIP with hashes and upstream notices, then tests the extracted bundle
on software Vulkan. It bundles a private Python 3.12 runtime with locked inference
dependencies and tests that package from stable and nightly Slicer. The ORT core
comes from the pinned upstream wheel; the patched native provider and host bridge
are compiled separately.

A second source build compiles ORT's C++ WebGPU provider to WebAssembly with
Emscripten. SlicerWeb receives that binary, matching JavaScript bindings and the
shared Python package in its extension wheel. Its published Slicer runtime is
reused. Both hosts run shared Python preparation and deterministic GPU fixtures;
the browser selects its own WebGPU implementation. These jobs use standard public
GitHub-hosted runners and explicitly identify software GPU adapters.

See [inference packaging](deployment/inference/README.md) and
[MONAI service packaging](deployment/monai/README.md) for build and run commands.
The extension archive, server containers, Python wheels and native asset ZIP are
separate deliverables. The desktop archive carries its private dependencies;
SlicerWeb uses Pyodide's NumPy/SciPy and the compiled WASM component. Application
code remains Python. WGSL remains in use; the proposed PySlang migration is later
work.

`deployment/migration-source.json` records the selected initial source snapshot.
Research and private deployment files remain in the internal repository. The
synthetic fixture validates packaging and transport, not trained-model quality.
Physical GPU, full workflow and extension-installation checks remain release gates.

## Development and publication

GitLab is the authoritative development repository. Its CI pipeline runs the
publication tests before pushing the tested default-branch commit or a protected
version tag to GitHub. Feature branches stay in GitLab. GitHub receives ordinary
Git commits with the same IDs, and published history is never force-pushed.

Run the infrastructure tests with Python 3.12+, Git, CMake 3.28+ and Make installed:

```sh
python3 -m unittest discover -s tests/infrastructure -v
```

The tests create temporary local Git repositories. They need no credentials or
network access.

## GitLab setup

Set `main` as the protected default branch and protect version tags matching `v*`.
Configure these protected file variables with environment scope `public`:

- `GITHUB_SSH_PRIVATE_KEY`: a dedicated SSH deploy key with write access to this
  GitHub repository only.
- `GITHUB_SSH_KNOWN_HOSTS`: GitHub's SSH host public keys, obtained from its
  authenticated HTTPS metadata API.

The pipeline uses these files directly without printing their contents. Publication
requires a push pipeline on a protected ref. Merge request, feature-branch and web
pipelines run tests only. Version tags must point to commits on the GitLab `main`
history. Create an annotated release tag such as `v1.0.0` in GitLab after the
corresponding default-branch pipeline succeeds.

This pipeline publishes the dedicated extension repository's history. Keep every
file and commit in this repository suitable for public release. Research sources,
private deployment settings, model weights and experimental data belong outside
this repository.

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
