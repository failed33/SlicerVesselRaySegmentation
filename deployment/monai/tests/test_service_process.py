"""Real HTTP and native startup with explicit Python or Docker test composition."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from requests_toolbelt.multipart.decoder import MultipartDecoder


@pytest.mark.parametrize(
    ("worker_mode", "outcome"),
    [
        ("python", "success"),
        ("python", "timeout"),
        ("container", "success"),
        ("container", "timeout"),
        ("container", "launch-failure"),
    ],
)
def test_http_worker_lifecycle(
    tmp_path: Path, worker_mode: str, outcome: str, live_service
) -> None:
    if worker_mode == "container" and not os.environ.get("MONAI_SMOKE_IMAGE"):
        pytest.skip("Local smoke image required")
    environment = {
        **os.environ,
        "MONAI_TEST_WORKER_IMAGE": os.environ["MONAI_SMOKE_IMAGE"]
        if worker_mode == "container"
        else "",
        "MONAI_SERVICE_WORK_DIR": str(tmp_path / "service"),
        "MONAI_TEST_WORKER_TIMEOUT_SECONDS": "0.001" if outcome == "timeout" else "30",
        "MONAI_LABEL_AUTH_ENABLE": "false",
        "MONAI_LABEL_DATASTORE_AUTO_RELOAD": "false",
    }
    if outcome == "launch-failure":
        environment["MONAI_SERVICE_CONTAINER_RUNTIME"] = str(tmp_path / "missing-runtime")
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5) - 28
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 1.1, 2.3))
    image.SetOrigin((20, -40, 60))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    input_path = tmp_path / "ct.nii.gz"
    sitk.WriteImage(image, str(input_path))
    with live_service(environment, app="tests.asgi:app") as client:
        response = client.post(
            "/infer/smoke-test", files={"file": ("ct.nii.gz", input_path.read_bytes())}
        )
        expected = {"success": 200, "timeout": 504, "launch-failure": 502}[outcome]
        assert response.status_code == expected, response.text
        if outcome == "success":
            parts = {
                part.headers[b"Content-Disposition"].split(b'name="')[1].split(b'"')[0]: part
                for part in MultipartDecoder.from_response(response).parts
            }
            result_path = tmp_path / "result.nii.gz"
            result_path.write_bytes(parts[b"image"].content)
            result = sitk.ReadImage(str(result_path))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(result), array > 0)
            assert result.GetDirection() == pytest.approx(image.GetDirection())
            assert result.GetSpacing() == pytest.approx(image.GetSpacing())
            assert result.GetOrigin() == pytest.approx(image.GetOrigin())
            assert json.loads(parts[b"params"].content)["test_only"] is True
        assert client.get("/info/").status_code == 200
        # Response background cleanup can finish just after the client reads the body.
        for _ in range(100):
            if not list((tmp_path / "service" / "requests").iterdir()):
                break
            time.sleep(0.01)
        assert not list((tmp_path / "service" / "requests").iterdir())
