# MONAI Label service

This service exposes ARG and ModelO through native MONAI Label discovery and
inference routes. Both tasks use `vessel-inference`. Direct execution is the
default; optional process, Docker and Kubernetes executors call the same workers.
Training, model publication and private deployment configuration remain outside
this repository.

## Build and install

From the repository root:

```sh
uv build deployment/inference --out-dir build/backend/dist
uv build deployment/monai --out-dir build/backend/dist
uv venv --python 3.12 build/backend/installed
uv pip install --python build/backend/installed/bin/python \
  --find-links build/backend/dist 'monai-label-service[arg]'
```

On Linux, preinstall matching CPU Torch and torchvision from the official
PyTorch CPU index when accelerator Torch is unnecessary. MONAI Label requires
Torch; the separate ARG ONNX worker does not.

Configure the service using your own model registry:

```sh
export MONAI_SERVICE_ENABLE_ARG=true
export MONAI_SERVICE_MLFLOW_TRACKING_URI=https://your-mlflow-server
export MONAI_SERVICE_ARG_MODEL_NAME=your-registered-model
export MONAI_SERVICE_ARG_MODEL_ALIAS=champion
export MONAI_SERVICE_WORK_DIR=/absolute/writable/service-directory
build/backend/installed/bin/python -m uvicorn asgi:app --host 127.0.0.1 --port 8000
```

The wheel includes `asgi.py`, `vessel_deployment.py`, `label_app/main.py` and
`monai_service`. Standard `python -m monailabel.main start_server --app <label_app>
--studies <studies>` discovery is also supported. Keep the app directory writable
for native Label's logs and auxiliary files. Use a persistent writable work
and session directory. Configure authentication and TLS for external hosting.

ModelO defaults to Dawn and needs the `model-o` extra and the separately built
matched native bundle. `MONAI_SERVICE_MODEL_O_ASSETS` points to its extracted
location. Model weights are supplied separately.

## CUDA execution

The CUDA environment supports both ARG and ModelO, including NVIDIA MIG devices.
It needs a CUDA 13 compatible driver, Python 3.12 and uv 0.10.7 or newer:

```sh
uv sync --project deployment/monai --python 3.12 --frozen --no-dev --group model-o-cuda
export MONAI_SERVICE_ARG_DEVICE=gpu
export MONAI_SERVICE_MODEL_O_BACKEND=cuda
uv run --project deployment/monai --no-sync python -m uvicorn \
  asgi:app --app-dir deployment/monai --host 127.0.0.1 --port 8000
```

Configure enabled models and their artifact sources before starting the server.
CUDA ModelO does not need `MONAI_SERVICE_MODEL_O_ASSETS`, Dawn or Vulkan. Its
sampler is generated from the same Slang source as the native/browser WebGPU
sampler. NVIDIA CUDA Python compiles it with NVRTC for the allocated GPU; ORT
owns the tensors and executes on the same CUDA stream. No training framework is
used by either inference worker. Label retains upstream's CPU Torch dependency.

The `model_o_cuda` container target supplies this worker environment. Select
`MONAI_SERVICE_MODEL_O_BACKEND=cuda` in the coordinator so its worker invocation
uses the matching backend. The existing `model_o` target supplies Dawn.

## Local model files

Serve downloaded MLflow artifacts without registry access on the server:

```sh
export MONAI_SERVICE_ARG_MODEL_DIR=/absolute/models/arg-onnx/1
export MONAI_SERVICE_MODEL_O_MODEL_DIR=/absolute/models/model-o/1
```

Use `<registered-model-name>/<MLflow-version>` directories containing the ordinary
serving files. The service derives the response's model name and version from
that path. These settings take precedence over registry lookup. A symlink to the
selected version can be used for updates.

## Containers

The build context is this repository, including both deployment packages:

```sh
docker build -f deployment/monai/Dockerfile.workers --target arg -t vessel-arg:local .
docker build -f deployment/monai/Dockerfile.workers --target label_arg -t vessel-label:local .
```

`arg` defaults to CPU. `label_arg` supports direct ARG inference and includes
MONAI Label's CPU Torch. `label` is the coordinator image for isolated workers;
`model_o` supplies ModelO's Python environment and consumes a mounted native
bundle. CUDA and MIGraphX targets need their corresponding runtime and hardware.
The official ARG WebGPU target remains experimental because its published
provider lacks the required ConvTranspose3D support.

When Label launches sibling containers, uploads, model cache and outputs must be
visible at the same absolute paths in each container. Set `TMPDIR` beneath that
shared mount. No Docker socket is needed for direct inference. For local bind mounts, build
Label with `--build-arg MONAI_UID="$(id -u)" --build-arg MONAI_GID="$(id -g)"`
and run it with `--user "$(id -u):$(id -g)"`. The named account and app directory
then match the host user; container caches remain removable by that user.

## Validation

`Backend CI` builds wheels from source distributions, installs them into a clean
environment, and starts the installed server away from the checkout. It registers
a generated input-dependent ONNX fixture in a temporary local MLflow registry,
then checks discovery, direct/process HTTP inference, exact mask voxels, oblique
image geometry, runtime metadata and request cleanup. It repeats HTTP inference
in the built `label_arg` container and runs the separate ARG container.

The standalone backend tests include authentication, validation, cancellation,
streaming, session lifetime and executor behavior. Desktop-client and cluster
manifest tests remain in the internal repository until those components migrate.
Synthetic inference proves packaging and integration. It does not establish
trained-model quality or accelerator parity.
