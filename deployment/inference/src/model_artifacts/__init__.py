"""Read-only MLflow model selection and complete-directory download reuse."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow.artifacts
from filelock import FileLock
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSource:
    """Server-controlled registry settings for one inference task."""

    name: str
    alias: str
    tracking_uri: str
    registry_uri: str | None = None

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.name, self.alias, self.tracking_uri)):
            raise ValueError("Model requires a name, alias and MLflow tracking URI")


@dataclass(frozen=True)
class SelectedModel:
    """An explicit version captured before worker startup."""

    name: str
    version: str
    tracking_uri: str
    registry_uri: str

    @property
    def artifact_uri(self) -> str:
        return f"models:/{self.name}/{self.version}"

    @property
    def cache_key(self) -> str:
        identity = (self.tracking_uri, self.registry_uri, self.name, self.version)
        return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()


def select_model(source: ModelSource) -> SelectedModel:
    """Resolve the alias once. Never register, promote or modify a model."""
    registry_uri = source.registry_uri or source.tracking_uri
    client = MlflowClient(tracking_uri=source.tracking_uri, registry_uri=registry_uri)
    version = client.get_model_version_by_alias(source.name, source.alias)
    return SelectedModel(source.name, str(version.version), source.tracking_uri, registry_uri)


def download_model(model: SelectedModel, cache_root: Path) -> Path:
    """Reuse a completed version; a failed download never becomes a cache entry."""
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / model.cache_key
    with FileLock(str(destination.with_suffix(".lock")), timeout=60):
        if destination.is_dir():
            return destination
        with TemporaryDirectory(prefix="download-", dir=cache_root) as temporary:
            # MLflow 3.16 resolves models:/ downloads through the global registry.
            client = MlflowClient(tracking_uri=model.tracking_uri, registry_uri=model.registry_uri)
            artifact_uri = client.get_model_version_download_uri(model.name, model.version)
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=artifact_uri,
                    dst_path=temporary,
                    tracking_uri=model.tracking_uri,
                    registry_uri=model.registry_uri,
                )
            ).resolve()
            if not downloaded.is_relative_to(Path(temporary).resolve()):
                raise ValueError("MLflow did not download the model into the requested directory")
            if not downloaded.is_dir() or not any(downloaded.iterdir()):
                raise ValueError("The model artifact must be a nonempty directory")
            downloaded.rename(destination)
    return destination
