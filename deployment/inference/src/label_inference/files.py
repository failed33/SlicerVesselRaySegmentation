"""Transfer request outputs to native Label response-cleanup ownership."""

import tempfile
from pathlib import Path


class ResultFileError(RuntimeError):
    """An inference result is missing or outside its request workspace."""


def _require_owned_result(source: Path, workspace: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ResultFileError("Workflow did not produce a result file")
    if not source.resolve().is_relative_to(workspace):
        raise ResultFileError("Workflow result must belong to its request workspace")


def retain_result(source: Path, workspace: Path, work_dir: Path) -> Path:
    """Transfer a request result to Label's response-cleanup ownership."""
    _require_owned_result(source, workspace)
    with tempfile.NamedTemporaryFile(
        prefix="inference-result-",
        suffix="".join(source.suffixes),
        dir=work_dir,
        delete=False,
    ) as stream:
        destination = Path(stream.name)
    try:
        source.replace(destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination
