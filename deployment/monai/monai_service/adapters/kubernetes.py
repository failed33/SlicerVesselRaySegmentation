"""Run portable worker invocations as namespaced Kubernetes Jobs."""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from monai_service.adapters.bindings import expand_arguments
from monai_service.adapters.kubernetes_files import StagedInvocation, stage_invocation
from monai_service.contracts import ExecutionError, ExecutionResult, ExecutionTimeout, Invocation
from monai_service.schemas.execution import KubernetesConfig

if TYPE_CHECKING:
    from kubernetes.client import BatchV1Api, CoreV1Api, V1Job, V1PodList

logger = logging.getLogger(__name__)
POLL_SECONDS = 1.0
API_TIMEOUT_SECONDS = 10.0
CLEANUP_TIMEOUT_SECONDS = 45.0
WORKER_LOG_TAIL_BYTES = 8192


class JobCleanupError(ExecutionError):
    """Job ownership could not be released; retain its staged files."""


@dataclass(frozen=True)
class KubernetesClients:
    batch: BatchV1Api
    core: CoreV1Api


@contextmanager
def _clients() -> Iterator[KubernetesClients]:
    try:
        from kubernetes import client, config
        from kubernetes.config.config_exception import ConfigException

        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()
        configuration = client.Configuration.get_default_copy()
        configuration.retries = 0
    except Exception as error:
        raise ExecutionError("Could not configure the Kubernetes worker client") from error
    with client.ApiClient(configuration) as api_client:
        yield KubernetesClients(client.BatchV1Api(api_client), client.CoreV1Api(api_client))


def _request_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExecutionTimeout("Kubernetes inference exceeded its execution deadline")
    return min(API_TIMEOUT_SECONDS, remaining)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ExecutionError("Kubernetes Job template requires object-valued spec sections")
    return value


def _worker(pod: dict[str, object]) -> dict[str, object]:
    containers = pod.get("containers")
    if not isinstance(containers, list):
        raise ExecutionError("Kubernetes Job template requires a containers list")
    workers = []
    for container in containers:
        worker = _object(container)
        if worker.get("name") == "worker":
            workers.append(worker)
    if len(workers) != 1:
        raise ExecutionError("Kubernetes Job template requires exactly one container named worker")
    return workers[0]


def _entries(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ExecutionError("Kubernetes Job volumes and volumeMounts must be lists")
    for entry in value:
        _object(entry)
    return cast(list[dict[str, object]], value)


def _binding_entries(
    entries: list[dict[str, object]], workspace_entries: list[dict[str, object]]
) -> list[dict[str, object]]:
    native_entries = [entry for entry in entries if entry.get("name") != "workspace"]
    return [*native_entries, *workspace_entries]


@dataclass(frozen=True)
class _JobTemplate:
    body: dict[str, object]
    metadata: dict[str, object]
    spec: dict[str, object]
    pod: dict[str, object]
    worker: dict[str, object]
    volumes: list[dict[str, object]]
    volume_mounts: list[dict[str, object]]

    def prepare(
        self, config: KubernetesConfig, invocation: Invocation, staged: StagedInvocation
    ) -> dict[str, object]:
        job = deepcopy(self)
        job.metadata.update(name=f"inference-{uuid.uuid4().hex}", namespace=config.namespace)
        job.spec["activeDeadlineSeconds"] = math.ceil(config.timeout_seconds)
        job.worker["args"] = expand_arguments(invocation, staged.paths)
        job.worker["volumeMounts"] = _binding_entries(job.volume_mounts, staged.mounts)
        job.pod["volumes"] = _binding_entries(
            job.volumes,
            [{"name": "workspace", "persistentVolumeClaim": {"claimName": config.pvc_claim}}],
        )
        return job.body


def _load_template(path: Path) -> _JobTemplate:
    import yaml

    try:
        template = _object(yaml.safe_load(path.read_text()))
        if template.get("apiVersion") != "batch/v1" or template.get("kind") != "Job":
            raise ExecutionError("Kubernetes worker template must be a batch/v1 Job")
        metadata = _object(template.setdefault("metadata", {}))
        spec = _object(template.get("spec"))
        pod = _object(_object(spec.get("template")).get("spec"))
        worker = _worker(pod)
        volumes = _entries(pod.setdefault("volumes", []))
        volume_mounts = _entries(worker.setdefault("volumeMounts", []))
        return _JobTemplate(template, metadata, spec, pod, worker, volumes, volume_mounts)
    except (OSError, yaml.YAMLError) as error:
        raise ExecutionError("Could not read Kubernetes worker Job template") from error


def _terminal_state(job: V1Job) -> str | None:
    if job.status is None:
        return None
    for condition in job.status.conditions or []:
        if condition.status == "True" and condition.type in {"Complete", "Failed"}:
            return condition.type
    return None


def _wait_for_job(
    clients: KubernetesClients, name: str, config: KubernetesConfig, deadline: float
) -> None:
    while True:
        job = cast(
            "V1Job",
            clients.batch.read_namespaced_job(
                name, config.namespace, _request_timeout=_request_timeout(deadline)
            ),
        )
        state = _terminal_state(job)
        if state == "Complete":
            return
        if state == "Failed":
            raise ExecutionError("Kubernetes inference Job failed")
        time.sleep(min(POLL_SECONDS, _request_timeout(deadline)))


def _log_failure(clients: KubernetesClients, name: str, namespace: str) -> None:
    deadline = time.monotonic() + API_TIMEOUT_SECONDS
    try:
        pods: V1PodList = cast(
            "V1PodList",
            clients.core.list_namespaced_pod(
                namespace,
                label_selector=f"batch.kubernetes.io/job-name={name}",
                _request_timeout=_request_timeout(deadline),
            ),
        )
        for pod in pods.items or []:
            log = clients.core.read_namespaced_pod_log(
                pod.metadata.name,
                namespace,
                container="worker",
                tail_lines=100,
                limit_bytes=WORKER_LOG_TAIL_BYTES,
                _request_timeout=_request_timeout(deadline),
            )
            logger.error("Kubernetes worker log: %s", log)
    except Exception:
        logger.exception("Could not read failed Kubernetes worker logs")


def _delete_job(clients: KubernetesClients, name: str, namespace: str) -> None:
    deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
    try:
        clients.batch.delete_namespaced_job(
            name,
            namespace,
            body={"propagationPolicy": "Foreground", "gracePeriodSeconds": 0},
            _request_timeout=_request_timeout(deadline),
        )
        while True:
            clients.batch.read_namespaced_job(
                name, namespace, _request_timeout=_request_timeout(deadline)
            )
            time.sleep(min(POLL_SECONDS, _request_timeout(deadline)))
    except Exception as error:
        if getattr(error, "status", None) == 404:
            return
        raise JobCleanupError("Could not clean up Kubernetes inference Job") from error


@dataclass(frozen=True)
class KubernetesExecutor:
    config: KubernetesConfig
    _template: _JobTemplate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_template", _load_template(self.config.job_template))

    def _run_job(
        self, clients: KubernetesClients, invocation: Invocation, staged: StagedInvocation
    ) -> str:
        job = self._template.prepare(self.config, invocation, staged)
        name = cast(dict[str, str], job["metadata"])["name"]
        deadline = time.monotonic() + self.config.timeout_seconds
        try:
            clients.batch.create_namespaced_job(
                self.config.namespace,
                job,
                _request_timeout=_request_timeout(deadline),
            )
            _wait_for_job(clients, name, self.config, deadline)
        except Exception as error:
            timed_out = time.monotonic() >= deadline
            _log_failure(clients, name, self.config.namespace)
            if isinstance(error, ExecutionError):
                raise
            if timed_out:
                raise ExecutionTimeout(
                    "Kubernetes inference exceeded its execution deadline"
                ) from error
            raise ExecutionError("Could not execute Kubernetes inference Job") from error
        finally:
            _delete_job(clients, name, self.config.namespace)
        return name

    def run(self, invocation: Invocation) -> ExecutionResult:
        staged = stage_invocation(invocation, self.config)
        cleaned_up = True
        try:
            with _clients() as clients:
                try:
                    name = self._run_job(clients, invocation, staged)
                except JobCleanupError:
                    cleaned_up = False
                    raise
                staged.retrieve()
            return ExecutionResult("kubernetes", f"{self.config.namespace}/{name}")
        except OSError as error:
            raise ExecutionError("Could not retrieve Kubernetes worker outputs") from error
        finally:
            if cleaned_up:
                staged.cleanup()
