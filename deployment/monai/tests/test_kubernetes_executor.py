"""Exercise Job lifecycle and PVC exchange without connecting to a cluster."""

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from monai_service.adapters import kubernetes as adapter
from monai_service.adapters.executor_config import executor_from_env
from monai_service.contracts import (
    ExecutionError,
    ExecutionTimeout,
    FileArgument,
    FileBinding,
    Invocation,
)
from monai_service.schemas.execution import KubernetesConfig


class ApiError(Exception):
    def __init__(self, status):
        self.status = status


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _job(state):
    conditions = [] if state is None else [SimpleNamespace(type=state, status="True")]
    return SimpleNamespace(status=SimpleNamespace(conditions=conditions))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    clock = Clock()
    monkeypatch.setattr(adapter.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(adapter.time, "sleep", clock.sleep)
    batch = Mock()
    core = Mock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    batch.read_namespaced_job.side_effect = [_job("Complete"), ApiError(404)]

    @contextmanager
    def clients():
        yield adapter.KubernetesClients(batch, core)

    monkeypatch.setattr(adapter, "_clients", clients)
    template = tmp_path / "job.yaml"
    template.write_text(
        json.dumps(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "spec": {
                    "backoffLimit": 0,
                    "ttlSecondsAfterFinished": 300,
                    "template": {
                        "metadata": {"annotations": {"linkerd.io/inject": "disabled"}},
                        "spec": {
                            "restartPolicy": "Never",
                            "automountServiceAccountToken": False,
                            "securityContext": {
                                "runAsUser": 1000,
                                "runAsGroup": 1000,
                                "fsGroup": 1000,
                            },
                            "containers": [
                                {
                                    "name": "worker",
                                    "image": "worker:version",
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                }
                            ],
                        },
                    },
                },
            }
        )
    )
    config = KubernetesConfig(
        mode="kubernetes",
        job_template=template,
        namespace="monai",
        pvc_claim="shared",
        shared_root=shared,
        timeout_seconds=3.0,
    )
    return SimpleNamespace(shared=shared, clock=clock, batch=batch, core=core, config=config)


def _invocation(root):
    image = root / "scan.nrrd"
    image.write_bytes(b"ct-values")
    output = root / "output"
    output.mkdir()
    return Invocation(
        arguments=(
            "--input",
            FileArgument("image"),
            "--output",
            FileArgument("output", PurePosixPath("mask.nii.gz")),
        ),
        bindings=(FileBinding("image", image), FileBinding("output", output, writable=True)),
    )


def _write_output(harness, namespace, body, **kwargs):
    mounts = body["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    output = next(mount for mount in mounts if mount["mountPath"].endswith("/output"))
    (harness.shared / output["subPath"] / "mask.nii.gz").write_bytes(b"mask")


def test_success_stages_external_upload_and_retrieves_output_before_cleaning(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.create_namespaced_job.side_effect = lambda *a, **kw: _write_output(
        harness, *a, **kw
    )
    result = adapter.KubernetesExecutor(harness.config).run(invocation)
    assert result.runtime == "kubernetes"
    assert result.reference.startswith("monai/inference-")
    assert (tmp_path / "output" / "mask.nii.gz").read_bytes() == b"mask"
    assert list(harness.shared.iterdir()) == []
    _, body = harness.batch.create_namespaced_job.call_args.args
    spec = body["spec"]
    assert spec["backoffLimit"] == 0
    assert spec["activeDeadlineSeconds"] == 3
    assert spec["ttlSecondsAfterFinished"] == 300
    assert spec["template"]["metadata"]["annotations"] == {"linkerd.io/inject": "disabled"}
    pod = spec["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["restartPolicy"] == "Never"
    container = pod["containers"][0]
    assert container["args"] == [
        "--input",
        "/bindings/image/scan.nrrd",
        "--output",
        "/bindings/output/mask.nii.gz",
    ]
    assert [mount["readOnly"] for mount in container["volumeMounts"]] == [True, False]
    assert all(mount["subPath"].startswith("job-") for mount in container["volumeMounts"])
    assert (
        harness.batch.delete_namespaced_job.call_args.kwargs["body"]["propagationPolicy"]
        == "Foreground"
    )


def test_shared_inputs_and_native_worker_policy_reach_job_unchanged(harness):
    invocation = _invocation(harness.shared)
    template = json.loads(harness.config.job_template.read_text())
    pod = template["spec"]["template"]["spec"]
    container = pod["containers"][0]
    pod.update(
        {
            "nodeSelector": {"kubernetes.io/arch": "amd64"},
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {"key": "gpu", "operator": "In", "values": ["amd"]}
                                ]
                            }
                        ]
                    }
                }
            },
            "serviceAccountName": "isolated-worker",
            "imagePullSecrets": [{"name": "registry"}],
            "tolerations": [
                {"key": "gpu", "operator": "Exists", "effect": "NoExecute", "tolerationSeconds": 5}
            ],
            "volumes": [{"name": "native-secret", "secret": {"secretName": "credential"}}],
        }
    )
    container.update(
        {
            "env": [
                {"name": "BACKEND", "value": "cpu"},
                {
                    "name": "TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "credential", "key": "token"}},
                },
            ],
            "envFrom": [{"secretRef": {"name": "runtime-env"}}],
            "resources": {"limits": {"amd.com/gpu": "1"}},
            "volumeMounts": [
                {"name": "native-secret", "mountPath": "/credential", "readOnly": True}
            ],
        }
    )
    template["spec"]["template"]["metadata"]["annotations"]["native-policy"] = "retained"
    harness.config.job_template.write_text(json.dumps(template))
    config = harness.config.model_copy(update={"mount_path": PurePosixPath("/worker")})
    harness.batch.create_namespaced_job.side_effect = lambda *a, **kw: _write_output(
        harness, *a, **kw
    )
    adapter.KubernetesExecutor(config).run(invocation)
    _, body = harness.batch.create_namespaced_job.call_args.args
    actual_pod = body["spec"]["template"]["spec"]
    actual_container = actual_pod["containers"][0]
    for name in [
        "nodeSelector",
        "affinity",
        "serviceAccountName",
        "imagePullSecrets",
        "tolerations",
        "securityContext",
    ]:
        assert actual_pod[name] == pod[name]
    for name in ["env", "envFrom", "resources", "securityContext"]:
        assert actual_container[name] == container[name]
    assert body["spec"]["template"]["metadata"] == template["spec"]["template"]["metadata"]
    assert actual_pod["volumes"][0] == pod["volumes"][0]
    assert actual_container["volumeMounts"][0] == container["volumeMounts"][0]
    assert actual_container["volumeMounts"][1]["subPath"] == "scan.nrrd"
    assert actual_container["volumeMounts"][2]["subPath"].startswith("job-")
    assert (harness.shared / "scan.nrrd").read_bytes() == b"ct-values"
    assert not list(harness.shared.glob("job-*"))


def test_failed_job_logs_namespace_scoped_worker_and_deletes_it(harness, tmp_path, caplog):
    invocation = _invocation(tmp_path)
    harness.batch.read_namespaced_job.side_effect = [_job("Failed"), ApiError(404)]
    harness.core.list_namespaced_pod.return_value = SimpleNamespace(
        items=[SimpleNamespace(metadata=SimpleNamespace(name="worker-pod"))]
    )
    harness.core.read_namespaced_pod_log.return_value = "worker rejected input"
    with pytest.raises(ExecutionError, match="Job failed"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    assert "worker rejected input" in caplog.text
    assert harness.core.list_namespaced_pod.call_args.args == ("monai",)
    assert harness.core.read_namespaced_pod_log.call_args.args == ("worker-pod", "monai")
    harness.batch.delete_namespaced_job.assert_called_once()
    assert not list(harness.shared.glob("job-*"))


def test_pending_job_uses_total_deadline_and_is_cleaned(harness, tmp_path):
    invocation = _invocation(tmp_path)
    reads = iter([_job(None)] * 3 + [ApiError(404)])
    harness.batch.read_namespaced_job.side_effect = lambda *a, **kw: _raise_or_return(next(reads))
    with pytest.raises(ExecutionTimeout):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    assert harness.clock.now == 3.0
    harness.batch.delete_namespaced_job.assert_called_once()
    assert not list(harness.shared.glob("job-*"))


def test_false_complete_condition_does_not_finish_the_request(harness, tmp_path):
    invocation = _invocation(tmp_path)
    incomplete = _job("Complete")
    incomplete.status.conditions[0].status = "False"
    harness.batch.read_namespaced_job.side_effect = [incomplete, _job("Complete"), ApiError(404)]
    adapter.KubernetesExecutor(harness.config).run(invocation)
    assert harness.clock.now == 1.0


def test_interrupted_worker_wait_still_removes_job_and_stage(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.read_namespaced_job.side_effect = [KeyboardInterrupt(), ApiError(404)]
    with pytest.raises(KeyboardInterrupt):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    harness.batch.delete_namespaced_job.assert_called_once()
    assert not list(harness.shared.glob("job-*"))


def test_failed_log_fetch_preserves_job_failure_and_cleanup(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.read_namespaced_job.side_effect = [_job("Failed"), ApiError(404)]
    harness.core.list_namespaced_pod.side_effect = ApiError(403)
    with pytest.raises(ExecutionError, match="Job failed"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    harness.batch.delete_namespaced_job.assert_called_once()
    assert not list(harness.shared.glob("job-*"))


def test_stage_is_retained_until_foreground_deletion_finishes(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.create_namespaced_job.side_effect = lambda *a, **kw: _write_output(
        harness, *a, **kw
    )
    reads = iter([_job("Complete"), _job(None), ApiError(404)])

    def read(*args, **kwargs):
        assert list(harness.shared.glob("job-*/output/mask.nii.gz"))
        assert not (tmp_path / "output" / "mask.nii.gz").exists()
        return _raise_or_return(next(reads))

    harness.batch.read_namespaced_job.side_effect = read
    adapter.KubernetesExecutor(harness.config).run(invocation)
    assert harness.clock.now == 1.0
    assert (tmp_path / "output" / "mask.nii.gz").read_bytes() == b"mask"
    assert not list(harness.shared.glob("job-*"))


def _raise_or_return(value):
    if isinstance(value, Exception):
        raise value
    return value


def test_creation_api_failure_still_deletes_uncertain_job(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.create_namespaced_job.side_effect = ApiError(500)
    harness.batch.delete_namespaced_job.side_effect = ApiError(404)
    with pytest.raises(ExecutionError, match="Could not execute"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    harness.batch.delete_namespaced_job.assert_called_once()
    assert not list(harness.shared.glob("job-*"))


def test_cleanup_failure_preserves_stage_and_does_not_publish_output(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.create_namespaced_job.side_effect = lambda *a, **kw: _write_output(
        harness, *a, **kw
    )
    harness.batch.delete_namespaced_job.side_effect = ApiError(403)
    with pytest.raises(adapter.JobCleanupError, match="clean up"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    assert list(harness.shared.glob("job-*/output/mask.nii.gz"))
    assert not (tmp_path / "output" / "mask.nii.gz").exists()


def test_foreground_cleanup_has_its_own_bounded_wait(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.batch.read_namespaced_job.side_effect = None
    harness.batch.read_namespaced_job.return_value = _job("Complete")
    with pytest.raises(adapter.JobCleanupError):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    assert harness.clock.now == adapter.CLEANUP_TIMEOUT_SECONDS
    assert list(harness.shared.glob("job-*"))


def test_client_configuration_failure_removes_stage(harness, tmp_path, monkeypatch):
    invocation = _invocation(tmp_path)

    def unavailable():
        raise ExecutionError("client unavailable")

    monkeypatch.setattr(adapter, "_clients", unavailable)
    with pytest.raises(ExecutionError, match="client unavailable"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    assert not list(harness.shared.glob("job-*"))


def test_missing_shared_mount_fails_before_job_creation(harness, tmp_path):
    invocation = _invocation(tmp_path)
    harness.shared.rmdir()
    with pytest.raises(ExecutionError, match="mounted directory"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    harness.batch.create_namespaced_job.assert_not_called()


def test_worker_output_symlink_is_rejected_before_retrieval(harness, tmp_path):
    invocation = _invocation(tmp_path)

    def produce_symlink(namespace, body, **kwargs):
        mounts = body["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        output = harness.shared / mounts[1]["subPath"]
        (output / "mask.nii.gz").symlink_to(tmp_path / "scan.nrrd")

    harness.batch.create_namespaced_job.side_effect = produce_symlink
    with pytest.raises(ExecutionError, match="symbolic links"):
        adapter.KubernetesExecutor(harness.config).run(invocation)
    assert not list(harness.shared.glob("job-*"))


def test_registry_factory_selects_kubernetes_without_loading_client(harness, monkeypatch):
    prefix = "MONAI_TEST_KUBE"
    values = {
        "MODE": "kubernetes",
        "JOB_TEMPLATE": str(harness.config.job_template),
        "NAMESPACE": "monai",
        "PVC_CLAIM": "shared",
        "SHARED_ROOT": str(harness.shared),
    }
    for field, value in values.items():
        monkeypatch.setenv(f"{prefix}_WORKER__{field}", value)
    assert isinstance(executor_from_env(prefix), adapter.KubernetesExecutor)


@pytest.mark.parametrize(
    "containers",
    [[], [{"name": "other", "image": "worker"}], [{"name": "worker"}, {"name": "worker"}]],
)
def test_invalid_named_worker_fails_when_loading_template_before_staging(harness, containers):
    template = json.loads(harness.config.job_template.read_text())
    template["spec"]["template"]["spec"]["containers"] = containers
    harness.config.job_template.write_text(json.dumps(template))
    with pytest.raises(ExecutionError, match="exactly one container named worker"):
        adapter.KubernetesExecutor(harness.config)
    assert not list(harness.shared.glob("job-*"))
    harness.batch.create_namespaced_job.assert_not_called()


@pytest.mark.parametrize(
    "content", ["bad: [yaml", "null", '{"apiVersion":"apps/v1","kind":"Deployment"}']
)
def test_malformed_or_non_job_template_fails_at_configuration(harness, content):
    harness.config.job_template.write_text(content)
    with pytest.raises(ExecutionError):
        adapter.KubernetesExecutor(harness.config)
    harness.batch.create_namespaced_job.assert_not_called()


@pytest.mark.parametrize(
    "sections, value",
    [
        (("metadata",), None),
        (("spec",), []),
        (("spec", "template"), None),
        (("spec", "template", "spec"), []),
        (("spec", "template", "spec", "containers"), {}),
        (("spec", "template", "spec", "containers"), [None]),
        (("spec", "template", "spec", "volumes"), {}),
        (("spec", "template", "spec", "volumes"), [None]),
    ],
)
def test_invalid_template_sections_fail_before_staging(harness, sections, value):
    template = json.loads(harness.config.job_template.read_text())
    section = template
    for name in sections[:-1]:
        section = section[name]
    section[sections[-1]] = value
    harness.config.job_template.write_text(json.dumps(template))
    with pytest.raises(ExecutionError):
        adapter.KubernetesExecutor(harness.config)
    assert not list(harness.shared.glob("job-*"))
    harness.batch.create_namespaced_job.assert_not_called()


@pytest.mark.parametrize("mounts", [{}, [None]])
def test_invalid_worker_mounts_fail_before_staging(harness, mounts):
    template = json.loads(harness.config.job_template.read_text())
    template["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = mounts
    harness.config.job_template.write_text(json.dumps(template))
    with pytest.raises(ExecutionError):
        adapter.KubernetesExecutor(harness.config)
    assert not list(harness.shared.glob("job-*"))
    harness.batch.create_namespaced_job.assert_not_called()


def test_yaml_section_aliases_are_preserved_without_mutating_loaded_policy(harness, tmp_path):
    invocation = _invocation(tmp_path)
    template = json.loads(harness.config.job_template.read_text())
    metadata = {"annotations": {"native-policy": "retained"}}
    template["metadata"] = metadata
    template["spec"]["template"]["metadata"] = metadata
    harness.config.job_template.write_text(yaml.safe_dump(template))
    executor = adapter.KubernetesExecutor(harness.config)
    executor.run(invocation)
    _, body = harness.batch.create_namespaced_job.call_args.args
    assert body["metadata"] is body["spec"]["template"]["metadata"]
    assert body["metadata"]["name"].startswith("inference-")
    assert body["metadata"]["namespace"] == "monai"
    assert body["metadata"]["annotations"] == metadata["annotations"]
    assert executor._template.body["metadata"] == metadata


def test_request_bindings_replace_only_named_workspace_entries(harness, tmp_path):
    invocation = _invocation(tmp_path)
    template = json.loads(harness.config.job_template.read_text())
    pod = template["spec"]["template"]["spec"]
    native_volume = {"name": "scratch", "emptyDir": {"medium": "Memory"}}
    native_mount = {"name": "scratch", "mountPath": "/scratch"}
    pod["volumes"] = [
        {"name": "workspace", "persistentVolumeClaim": {"claimName": "old-claim"}},
        native_volume,
    ]
    pod["containers"][0]["volumeMounts"] = [
        {"name": "workspace", "mountPath": "/old-bindings"},
        native_mount,
    ]
    helper = {"name": "helper", "image": "helper-image", "volumeMounts": [native_mount]}
    pod["containers"].append(helper)
    harness.config.job_template.write_text(json.dumps(template))
    adapter.KubernetesExecutor(harness.config).run(invocation)
    _, body = harness.batch.create_namespaced_job.call_args.args
    actual_pod = body["spec"]["template"]["spec"]
    assert actual_pod["volumes"] == [
        native_volume,
        {"name": "workspace", "persistentVolumeClaim": {"claimName": "shared"}},
    ]
    mounts = actual_pod["containers"][0]["volumeMounts"]
    assert mounts[0] == native_mount
    assert [mount["mountPath"] for mount in mounts[1:]] == [
        "/bindings/image/scan.nrrd",
        "/bindings/output",
    ]
    assert actual_pod["containers"][1] == helper


def test_loaded_template_is_immutable_across_two_requests(harness, tmp_path):
    invocation = _invocation(tmp_path)
    original_file = harness.config.job_template.read_text()
    executor = adapter.KubernetesExecutor(harness.config)
    original_template = deepcopy(executor._template.body)
    harness.batch.create_namespaced_job.side_effect = lambda *a, **kw: _write_output(
        harness, *a, **kw
    )
    harness.batch.read_namespaced_job.side_effect = [
        _job("Complete"),
        ApiError(404),
        _job("Complete"),
        ApiError(404),
    ]
    first = executor.run(invocation)
    # Replacing the source file after configuration must not change the loaded policy.
    harness.config.job_template.write_text("not a job")
    second = executor.run(invocation)
    assert first.reference != second.reference
    assert executor._template.body == original_template
    assert not list(harness.shared.glob("job-*"))
    bodies = [call.args[1] for call in harness.batch.create_namespaced_job.call_args_list]
    assert bodies[0]["spec"]["template"]["spec"]["containers"][0]["image"] == "worker:version"
    assert bodies[1]["spec"]["template"]["spec"]["containers"][0]["image"] == "worker:version"
    assert (
        "volumeMounts" not in json.loads(original_file)["spec"]["template"]["spec"]["containers"][0]
    )


def test_named_worker_can_follow_another_native_container(harness, tmp_path):
    invocation = _invocation(tmp_path)
    template = json.loads(harness.config.job_template.read_text())
    other = {"name": "helper", "image": "helper-image", "command": ["true"]}
    template["spec"]["template"]["spec"]["containers"].insert(0, other)
    harness.config.job_template.write_text(json.dumps(template))
    adapter.KubernetesExecutor(harness.config).run(invocation)
    _, body = harness.batch.create_namespaced_job.call_args.args
    containers = body["spec"]["template"]["spec"]["containers"]
    assert containers[0] == other
    assert containers[1]["args"][0] == "--input"
    assert containers[1]["volumeMounts"]


@pytest.mark.parametrize("user", [0, 1000])
def test_worker_ownership_configuration_reaches_native_pod_security_context(
    harness, tmp_path, user
):
    invocation = _invocation(tmp_path)
    template = json.loads(harness.config.job_template.read_text())
    template["spec"]["template"]["spec"]["securityContext"] = {
        "runAsUser": user,
        "runAsGroup": user,
        "fsGroup": 1000,
    }
    harness.config.job_template.write_text(json.dumps(template))
    adapter.KubernetesExecutor(harness.config).run(invocation)
    _, body = harness.batch.create_namespaced_job.call_args.args
    pod = body["spec"]["template"]["spec"]
    assert pod["securityContext"] == {"runAsUser": user, "runAsGroup": user, "fsGroup": 1000}
    assert pod["containers"][0]["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }


def test_model_directory_and_parameter_file_are_available_on_shared_pvc(harness, tmp_path):
    invocation = _invocation(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"graph")
    parameters = tmp_path / "parameters.json"
    parameters.write_text('{"polylines_zyx": [[[1,2,3]]]}')
    invocation = Invocation(
        arguments=(
            *invocation.arguments,
            "--model",
            FileArgument("model", PurePosixPath("model.onnx")),
            "--parameters",
            FileArgument("parameters"),
        ),
        bindings=(
            *invocation.bindings,
            FileBinding("model", model),
            FileBinding("parameters", parameters),
        ),
    )

    def worker(namespace, body, **kwargs):
        container = body["spec"]["template"]["spec"]["containers"][0]
        mounts = container["volumeMounts"]
        assert (harness.shared / mounts[2]["subPath"] / "model.onnx").read_bytes() == b"graph"
        assert (harness.shared / mounts[3]["subPath"]).read_text() == parameters.read_text()
        assert mounts[2]["readOnly"] and mounts[3]["readOnly"]
        assert container["args"][-4:] == [
            "--model",
            "/bindings/model/model.onnx",
            "--parameters",
            "/bindings/parameters/parameters.json",
        ]
        _write_output(harness, namespace, body, **kwargs)

    harness.batch.create_namespaced_job.side_effect = worker
    adapter.KubernetesExecutor(harness.config).run(invocation)
    assert not list(harness.shared.glob("job-*"))


def test_writable_directory_retrieval_preserves_worker_deletions(harness, tmp_path):
    invocation = _invocation(tmp_path)
    (tmp_path / "output" / "obsolete.txt").write_text("old")

    def worker(namespace, body, **kwargs):
        output = (
            harness.shared
            / body["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][1]["subPath"]
        )
        (output / "obsolete.txt").unlink()
        _write_output(harness, namespace, body, **kwargs)

    harness.batch.create_namespaced_job.side_effect = worker
    adapter.KubernetesExecutor(harness.config).run(invocation)
    assert not (tmp_path / "output" / "obsolete.txt").exists()
    assert (tmp_path / "output" / "mask.nii.gz").read_bytes() == b"mask"


def test_official_client_serializes_job_and_deserializes_terminal_condition(
    harness, tmp_path, monkeypatch
):
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException
    from kubernetes.client.rest import RESTResponse
    from urllib3.response import HTTPResponse

    invocation = _invocation(tmp_path)
    requests = []
    state = {"job": None, "deleted": False}

    def request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        if method == "POST":
            state["job"] = kwargs["body"]
            _write_output(harness, "monai", kwargs["body"])
        if method == "DELETE":
            state["deleted"] = True
            value = {"apiVersion": "v1", "kind": "Status", "status": "Success"}
        elif state["deleted"]:
            raise ApiException(
                http_resp=RESTResponse(HTTPResponse(body=b'{"reason":"NotFound"}', status=404))
            )
        else:
            value = {
                **state["job"],
                "status": {"conditions": [{"type": "Complete", "status": "True"}]},
            }
        return RESTResponse(HTTPResponse(body=json.dumps(value).encode(), status=200))

    @contextmanager
    def clients():
        with client.ApiClient() as api_client:
            monkeypatch.setattr(api_client, "request", request)
            yield adapter.KubernetesClients(
                client.BatchV1Api(api_client), client.CoreV1Api(api_client)
            )

    monkeypatch.setattr(adapter, "_clients", clients)
    result = adapter.KubernetesExecutor(harness.config).run(invocation)
    assert result.runtime == "kubernetes"
    assert [method for method, _, _ in requests] == ["POST", "GET", "DELETE", "GET"]
    assert all("/apis/batch/v1/namespaces/monai/jobs" in url for _, url, _ in requests)
    assert requests[0][2]["body"]["spec"]["activeDeadlineSeconds"] == 3
    assert requests[2][2]["body"] == {"propagationPolicy": "Foreground", "gracePeriodSeconds": 0}
    assert (tmp_path / "output" / "mask.nii.gz").read_bytes() == b"mask"
