"""Serve a local MLflow model directory laid out as <model-name>/<version>."""

from dataclasses import dataclass
from pathlib import Path

from monai_service.contracts import ArtifactIdentity, ResolvedArtifact


@dataclass(frozen=True)
class DirectoryArtifactSource:
    directory: Path

    def resolve(self) -> ResolvedArtifact:
        directory = self.directory.expanduser().resolve()
        name, version = directory.parent.name, directory.name
        return ResolvedArtifact(
            ArtifactIdentity(name, version, f"models:/{name}/{version}"), directory
        )
