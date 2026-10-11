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

ModelO additionally needs the `model-o` extra and the separately built matched
native bundle. `MONAI_SERVICE_MODEL_O_ASSETS` points to its extracted
location. Model weights are resolved through MLflow, never baked into packages.

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
shared mount. No Docker socket is needed for direct inference.

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
