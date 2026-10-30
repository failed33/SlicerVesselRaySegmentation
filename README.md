# SlicerVesselRaySegmentation - Publication WIP

This repository contains the ... inference backend, MONAI Label
service, scripted module. Model files will be provided separately

The workflow is endpoint setup → ARG vessel inference → component/mass filtering and manual trimming → centerline generation and editing → ModelO inference.

Another part of this repository is to provide the runtime for local fully WebGPU based and WASM ready ONNX based inference for the nnU-Net and other ray based model. We also want to provide the MONAI Label inference server, that can be hosted on the same machine with native GPU  acceleration, including scheduling on a kubernetes cluster and ROCm GPU (RX 6600, RX 7900XT), DGX Spark and HPC environment with NVIDIA A100 and H100. The setup was tested for all of these use cases (and I dont have a consumer NVIDIA card ...).

The primary segmentation is based on TopBrain Challenge 2nd Place ARG Team and their model, which we finetuned on our internal and expanded dataset to not only include CoW anatomy, but also the full Head-Neck arterial vascular Anatomy.

The rest is AI generated slop, as I haven't cleaned it up.
---

## Repository layout

| Path | Contents |
| --- | --- |
| `VesselRaySegmentation/` | Production Slicer workflow and scene/client helpers |
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
runs on version tags, pull requests, manual dispatch and weekly
on Sunday at 04:27 UTC. It uses standard public GitHub-hosted Ubuntu runners:

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

These jobs validate packaging and real application integration with synthetic
volumes, native segmentation cleanup and editable Markups. Backend CI separately
checks compiled inference. Trained-model quality and patient workflow acceptance
remain separate gates.

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
code remains Python. Typed Slang generates the packaged WGSL and reflection;
ONNX graph preparation lowers normalization to centered axiswise FP32 standard
operations without a Torch dependency in the application.

`deployment/migration-source.json` records the selected initial source snapshot.
Research and private deployment files remain in the internal repository. The
synthetic fixture validates packaging and transport, not trained-model quality.
Physical GPU, full workflow and extension-installation checks remain release gates.

## Application workflow

Configure one MONAI Label endpoint and refresh its model catalogue. The module
uses the server's image and polyline input profiles to offer ARG and ModelO models.
Choose the CT and run ARG. Create a separate cleanup segmentation, retain the
wanted components by count and minimum physical volume, then use native Segment
Editor tools for trimming. The original ARG prediction is retained.

Automatic network extraction requires SlicerVMTK's ExtractCenterline module.
Review and edit the resulting native Markups curves before ModelO. Curves can
also be created manually or imported; the client reads their world coordinates
and preserves continuous positions on the submitted CT grid.

The current SlicerWeb runtime does not bundle VMTK. Its manual/imported-curve
path remains available; automatic extraction is explicitly unavailable until
that dependency is installed. Browser inference uses fetch and requires an
endpoint accessible over HTTPS with CORS permission for the application's origin.
Bearer tokens stay in the current widget and are not saved into scenes or settings.
Desktop MONAI requests use HTTPX. At first endpoint discovery, Slicer's native
package manager checks `VesselRaySegmentationLib/Resources/requirements.txt`
and offers to install missing dependencies. SlicerWeb uses browser Fetch and
FormData and does not load HTTPX. Both transports retain native Label image
sessions and reject redirects.

The workflow acknowledges request preparation, image upload, session creation or
reuse, waiting for inference, and result import. Desktop uploads show encoded
request bytes and their average rate; the session response confirms receipt.
Browser Fetch has no upload-byte callback and shows activity while uploading.
While inference is pending, a busy indicator and elapsed counter stay visible.
An endpoint check runs every 15 seconds with a five-second timeout. Its result
indicates endpoint reachability, not model advancement. There is no model
percentage or estimated completion time. Discarding a result does not cancel
remote inference; controls unlock when the request settles or times out.

`slicerray-serve` emits a few request lifecycle records for operators. Worker
diagnostics remain available on failure; routine inference does not forward
per-patch output into the UI.


## Development and publication

GitLab is the authoritative development repository. Every branch and merge request
runs Ruff, the three package-specific Pyrefly scopes, CPU inference and service
tests, workflow transport tests, and publication/packaging tests. Production and
CI changes also run shader regeneration, installed numerical-wheel checks,
backend wheel/container acceptance, native and WASM source builds, and packaged
inference in stable Slicer, nightly Slicer and SlicerWeb. These checks always run
on default-branch and tag pipelines, and all must pass before publication to GitHub.
The two full package-check jobs use the `linux, large` runner class; lint and
publication use the shared `linux` tag. Acceptance uses the `linux, large, amd64`
runner class. Dependency downloads and bounded compiler caches are reusable;
compiled runtime bundles pass between jobs as one-day pipeline artifacts.
GitHub retains weekly, public pull-request, version-tag and manual application
and backend coverage. Its Python quality checks also run on `main` publications.
Feature branches stay in GitLab. GitHub receives ordinary
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
