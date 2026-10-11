"""Model selection is read-only and failed downloads cannot poison reuse."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import model_artifacts as models
import pytest


def selected(
    version: str = "1", registry: str = "https://registry.example"
) -> models.SelectedModel:
    return models.SelectedModel("arg_vessel", version, "https://tracking.example", registry)


@pytest.fixture
def model_download_client(monkeypatch):
    class Client:
        def __init__(self, **_kwargs):
            pass

        def get_model_version_download_uri(self, name, version):
            return f"file:///artifacts/{name}/{version}"

    monkeypatch.setattr(models, "MlflowClient", Client)


def test_alias_is_resolved_once_before_download(monkeypatch, tmp_path):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_model_version_by_alias(self, name, alias):
            calls.append((name, alias))
            return SimpleNamespace(version="12")

        def get_model_version_download_uri(self, name, version):
            calls.append(("download", name, version))
            return f"file:///artifacts/{name}/{version}"

    monkeypatch.setattr(models, "MlflowClient", Client)
    source = models.ModelSource("arg_vessel", "champion", "https://tracking.example")
    model = models.select_model(source)
    assert model.version == "12"
    assert calls == [
        {"tracking_uri": source.tracking_uri, "registry_uri": source.tracking_uri},
        ("arg_vessel", "champion"),
    ]
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs["artifact_uri"])
        output = Path(kwargs["dst_path"]) / "artifact"
        output.mkdir()
        (output / "weights").write_bytes(b"version twelve")
        return str(output)

    monkeypatch.setattr(models.mlflow.artifacts, "download_artifacts", download)
    path = models.download_model(model, tmp_path)
    assert (path / "weights").read_bytes() == b"version twelve"
    assert downloads == ["file:///artifacts/arg_vessel/12"]
    assert calls == [
        {"tracking_uri": source.tracking_uri, "registry_uri": source.tracking_uri},
        ("arg_vessel", "champion"),
        {"tracking_uri": source.tracking_uri, "registry_uri": source.tracking_uri},
        ("download", "arg_vessel", "12"),
    ]


@pytest.mark.usefixtures("model_download_client")
def test_complete_download_is_reused_and_versions_stay_separate(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs["artifact_uri"])
        output = Path(kwargs["dst_path"]) / "artifact"
        output.mkdir()
        (output / "weights").write_text(kwargs["artifact_uri"])
        return str(output)

    monkeypatch.setattr(models.mlflow.artifacts, "download_artifacts", download)
    first = models.download_model(selected("1"), tmp_path)
    assert models.download_model(selected("1"), tmp_path) == first
    second = models.download_model(selected("2"), tmp_path)
    assert first != second
    assert (first / "weights").read_text() == "file:///artifacts/arg_vessel/1"
    assert (second / "weights").read_text() == "file:///artifacts/arg_vessel/2"
    assert calls == ["file:///artifacts/arg_vessel/1", "file:///artifacts/arg_vessel/2"]
    assert selected(registry="https://another.example").cache_key != selected().cache_key


@pytest.mark.usefixtures("model_download_client")
def test_failed_download_is_cleaned_and_can_be_retried(monkeypatch, tmp_path):
    attempts = 0

    def download(**kwargs):
        nonlocal attempts
        attempts += 1
        output = Path(kwargs["dst_path"]) / "artifact"
        output.mkdir()
        (output / "weights").write_bytes(b"partial" if attempts == 1 else b"complete")
        if attempts == 1:
            raise OSError("connection interrupted")
        return str(output)

    monkeypatch.setattr(models.mlflow.artifacts, "download_artifacts", download)
    with pytest.raises(OSError, match="interrupted"):
        models.download_model(selected(), tmp_path)
    assert not (tmp_path / selected().cache_key).exists()
    assert not list(tmp_path.glob("download-*"))
    assert (models.download_model(selected(), tmp_path) / "weights").read_bytes() == b"complete"


@pytest.mark.usefixtures("model_download_client")
def test_simultaneous_misses_publish_one_complete_directory(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs["artifact_uri"])
        output = Path(kwargs["dst_path"]) / "artifact"
        output.mkdir()
        (output / "weights").write_bytes(b"complete")
        return str(output)

    monkeypatch.setattr(models.mlflow.artifacts, "download_artifacts", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(models.download_model, selected(), tmp_path) for _ in range(2)]
        paths = [future.result() for future in futures]
    assert paths[0] == paths[1]
    assert (paths[0] / "weights").read_bytes() == b"complete"
    assert calls == ["file:///artifacts/arg_vessel/1"]


@pytest.mark.usefixtures("model_download_client")
def test_download_cannot_cache_unrelated_directory(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weights").write_bytes(b"keep")
    monkeypatch.setattr(
        models.mlflow.artifacts, "download_artifacts", lambda **kwargs: str(outside)
    )
    with pytest.raises(ValueError, match="requested directory"):
        models.download_model(selected(), tmp_path / "cache")
    assert (outside / "weights").read_bytes() == b"keep"


def test_local_registry_download_preserves_selection_and_ignores_global_registry(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    registry_uri = f"sqlite:///{tmp_path / 'registry.db'}"
    decoy_uri = f"sqlite:///{tmp_path / 'decoy.db'}"
    monkeypatch.setattr(models.mlflow, "get_tracking_uri", lambda: decoy_uri)
    monkeypatch.setattr(models.mlflow, "get_registry_uri", lambda: decoy_uri)
    name = "arg_vessel"
    clients = {
        "selected": models.MlflowClient(tracking_uri=registry_uri, registry_uri=registry_uri),
        "decoy": models.MlflowClient(tracking_uri=decoy_uri, registry_uri=decoy_uri),
    }
    for client in clients.values():
        client.create_registered_model(name)

    sources = {}
    for label, client, contents in (
        ("first", clients["selected"], b"selected version one"),
        ("second", clients["selected"], b"selected version two"),
        ("decoy", clients["decoy"], b"wrong registry"),
    ):
        artifact = tmp_path / label
        artifact.mkdir()
        (artifact / "checkpoint_best.pth").write_bytes(contents)
        (artifact / "plans.json").write_text("{}")
        (artifact / "dataset.json").write_text("{}")
        client.create_model_version(name, artifact.as_uri())
        sources[label] = artifact

    client = clients["selected"]
    client.set_registered_model_alias(name, "champion", "1")
    source = models.ModelSource(name, "champion", registry_uri)
    first = models.select_model(source)
    client.set_registered_model_alias(name, "champion", "2")

    cache_root = tmp_path / "cache"
    first_path = models.download_model(first, cache_root)
    assert (first_path / "checkpoint_best.pth").read_bytes() == b"selected version one"
    second = models.select_model(source)
    second_path = models.download_model(second, cache_root)
    assert first.version == "1"
    assert second.version == "2"
    assert first_path != second_path
    assert (second_path / "checkpoint_best.pth").read_bytes() == b"selected version two"
    assert (first_path / "plans.json").is_file()
    assert (first_path / "dataset.json").is_file()
    assert not (first_path / "MLmodel").exists()
    assert not list(cache_root.glob("download-*"))

    def unexpected_download(**_kwargs):
        raise AssertionError("A completed model must be reused without downloading")

    monkeypatch.setattr(models.mlflow.artifacts, "download_artifacts", unexpected_download)
    assert models.download_model(first, cache_root) == first_path
    assert models.download_model(second, cache_root) == second_path
    assert (sources["first"] / "checkpoint_best.pth").read_bytes() == b"selected version one"
    assert (sources["second"] / "checkpoint_best.pth").read_bytes() == b"selected version two"
    assert (sources["decoy"] / "checkpoint_best.pth").read_bytes() == b"wrong registry"
