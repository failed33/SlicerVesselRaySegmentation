"""Resolve locally synchronized model snapshots without contacting MLflow."""

from dataclasses import dataclass
from pathlib import Path

from model_artifacts.snapshot import read_snapshot

from monai_service.contracts import ArtifactIdentity, ArtifactUnavailable, ResolvedArtifact


@dataclass(frozen=True)
class DirectoryArtifactSource:
    directory: Path

    def resolve(self) -> ResolvedArtifact:
        try:
            # Resolve a deployment's current symlink once per request. The
            # worker retains this immutable version while later syncs advance it.
            directory = self.directory.expanduser().resolve(strict=True)
            manifest = read_snapshot(directory)
            model = manifest["model"]
            return ResolvedArtifact(
                ArtifactIdentity(model["name"], model["version"], model["uri"]), directory
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ArtifactUnavailable("Configured local model snapshot is unavailable") from error
