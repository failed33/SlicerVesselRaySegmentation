"""Adapt shared MLflow artifact retrieval to application contracts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from model_artifacts import ModelSource, download_model, select_model

from monai_service.contracts import ArtifactIdentity, ArtifactUnavailable, ResolvedArtifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLflowArtifactSource:
    config: ModelSource
    cache_root: Path

    def resolve(self) -> ResolvedArtifact:
        try:
            selected = select_model(self.config)
            path = download_model(selected, self.cache_root)
        except Exception as error:
            logger.exception("Could not resolve or download configured model %s", self.config.name)
            raise ArtifactUnavailable("Configured model artifact is unavailable") from error
        return ResolvedArtifact(
            ArtifactIdentity(selected.name, selected.version, selected.artifact_uri), path
        )
