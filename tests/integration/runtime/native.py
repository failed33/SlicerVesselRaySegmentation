"""Accept installed application code and native ORT/Dawn with shader-owned inputs."""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
from application import application_summary
from dawn_ort import DawnHost, DeviceTensor, GpuSession
from dawn_ort.platforms import platform_config
from model_o import run_native as run_model_o


def adapter_info(host: DawnHost, allow_software: bool) -> dict[str, object]:
    w = host.w
    info = w.WGPUAdapterInfo()
    if w.wgpuAdapterGetInfo(host.adapter, ct.byref(info)) != w.WGPUStatus_Success:
        raise RuntimeError("Dawn could not query its actual adapter")
    try:
        assert info.backendType == getattr(w, f"WGPUBackendType_{host.backend}")
        assert info.backendType != w.WGPUBackendType_Null
        software = info.adapterType == w.WGPUAdapterType_CPU
        assert allow_software or not software, "Software adapter requires --allow-software"
        return {
            "backend": host.backend,
            "software": software,
            "description": host._message(info.description),
            "vendor": host._message(info.vendor),
            "device": host._message(info.device),
        }
    finally:
        w.wgpuAdapterInfoFreeMembers(info)


def run_case(host: DawnHost, fixtures: Path, spec: dict[str, Any]) -> dict[str, object]:
    model = (fixtures / spec["model"]).read_bytes()
    assert hashlib.sha256(model).hexdigest() == spec["sha256"], "Model fixture hash changed"
    dtype = np.dtype(spec["dtype"])
    with ExitStack() as resources:
        x = host.buffer(spec["input_elements"] * dtype.itemsize)
        y = host.buffer(spec["output_elements"] * dtype.itemsize)
        audit = host.buffer(spec["output_elements"] * spec["iterations"] * 4)
        for buffer in (x, y, audit):
            resources.callback(buffer.close)
        producer = host.kernel(spec["producer"], [x], entry_point="produce")
        consumer = host.kernel(spec["consumer"], [y, x, audit], entry_point="consume")
        resources.callback(producer.close)
        resources.callback(consumer.close)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(fixtures / spec["name"])
        session = GpuSession(
            host, model, options=options, provider_options={"preferredLayout": "NCHW"}
        )
        resources.callback(session.close)
        session.native_session.disable_fallback()
        session.bind(
            {"x": DeviceTensor(x, tuple(spec["input_shape"]), dtype)},
            outputs={"y": DeviceTensor(y, tuple(spec["output_shape"]), dtype)},
        )
        providers = session.native_session.get_providers()
        assert "WebGpuExecutionProvider" in providers
        # These counters observe DawnHost's Python calls, not ORT/Dawn internals.
        before = (host.wait_count, host.write_bytes, host.readback_count)
        for _ in range(spec["iterations"]):
            producer.dispatch((spec["input_elements"] + 63) // 64)
            outputs = session.submit()
            assert outputs["y"].owner.address == y.address, "ORT replaced the caller-owned output"
            consumer.dispatch((spec["output_elements"] + 63) // 64)
        assert host.wait_count == before[0], "Intermediate GPU completion wait"
        assert host.write_bytes == before[1], "Shader-owned input was uploaded from the CPU"
        actual = host.read(audit)
        assert host.readback_count - before[2] == 1
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, spec["expected"], rtol=spec["rtol"], atol=spec["atol"])
        assert not np.array_equal(
            actual[: spec["output_elements"]], actual[-spec["output_elements"] :]
        )
        host.raise_errors()
        profile = json.loads(Path(session.native_session.end_profiling()).read_text())
        assignments = [
            event["args"]["provider"]
            for event in profile
            if event.get("args", {}).get("op_name") == "ConvTranspose"
        ]
        assert assignments == ["WebGpuExecutionProvider"] * spec["iterations"]
        return {
            "name": spec["name"],
            "status": "passed",
            "model_sha256": spec["sha256"],
            "dtype": spec["dtype"],
            "input_shape": spec["input_shape"],
            "output_shape": spec["output_shape"],
            "iterations": spec["iterations"],
            "providers": providers,
            "node_providers": assignments,
            "cpu_fallback_disabled": True,
            "caller_output_buffer_retained": True,
            "caller_completion_waits": 0,
            "caller_input_upload_bytes": 0,
            "caller_final_readbacks": 1,
            "values": actual.tolist(),
        }


def run(fixtures: Path, assets: Path, *, allow_software: bool, require_f16: bool) -> dict[str, Any]:
    fixtures, assets = fixtures.resolve(), assets.resolve()
    if allow_software:
        os.environ["ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER"] = "1"
    cases = json.loads((fixtures / "cases.json").read_text())
    assert len(cases) == 4
    with ExitStack() as resources:
        ort.register_execution_provider_library(
            "runtime_acceptance", str(assets / platform_config(sys.platform).provider_library)
        )
        resources.callback(ort.unregister_execution_provider_library, "runtime_acceptance")
        host = DawnHost(assets / "webgpu.py")
        resources.callback(host.close)
        adapter = adapter_info(host, allow_software)
        supports_f16 = bool(
            host.w.wgpuDeviceHasFeature(host.device, host.w.WGPUFeatureName_ShaderF16)
        )
        assert supports_f16 or not require_f16, "Device lacks shader-f16"
        results = []
        for lifecycle in range(2):
            for spec in cases:
                if spec["dtype"] == "float16" and not supports_f16:
                    result = {
                        "name": spec["name"],
                        "status": "skipped",
                        "reason": "shader-f16 unavailable",
                    }
                else:
                    result = run_case(host, fixtures, spec)
                results.append({"lifecycle": lifecycle, **result})
        return {
            "status": "passed",
            "ort_version": ort.__version__,
            "adapter": adapter,
            "shader_f16": supports_f16,
            "physical_gpu_acceptance": not adapter["software"],
            "observer_scope": {
                "completion_waits": "DawnHost Python wait counter only; excludes ORT/Dawn internals.",
                "input_uploads": "DawnHost Python write counter only; excludes ORT/Dawn internals.",
                "readbacks": "DawnHost Python read counter only; excludes ORT/Dawn internals.",
            },
            "application": application_summary(),
            "model_o_shaders": run_model_o(host, fixtures / "model_o"),
            "cases": results,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--native-assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-software", action="store_true")
    parser.add_argument("--require-f16", action="store_true")
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    report = run(
        args.fixtures,
        args.native_assets,
        allow_software=args.allow_software,
        require_f16=args.require_f16,
    )
    if args.package_root is not None:
        import dawn_ort

        root = args.package_root.resolve()
        for loaded in (dawn_ort.__file__, report["application"]["module_path"]):
            assert Path(loaded).resolve().is_relative_to(root), (
                f"Import outside bundled environment: {loaded}"
            )
        report["package_root"] = str(root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "adapter": report["adapter"],
                "cases_passed": sum(case["status"] == "passed" for case in report["cases"]),
                "shader_f16": report["shader_f16"],
            }
        )
    )


if __name__ == "__main__":
    main()
