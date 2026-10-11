"""Relocate a native bundle and exercise its provider using shared Dawn buffers."""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from dawn_ort import DawnHost, DeviceTensor, GpuSession
from dawn_ort.platforms import platform_config


def check_provider(assets: Path) -> dict:
    import sys

    with ExitStack() as resources:
        ort.register_execution_provider_library(
            "bundle_acceptance", str(assets / platform_config(sys.platform).provider_library)
        )
        resources.callback(ort.unregister_execution_provider_library, "bundle_acceptance")
        host = DawnHost(assets / "webgpu.py")
        resources.callback(host.close)
        info = host.w.WGPUAdapterInfo()
        assert host.w.wgpuAdapterGetInfo(host.adapter, ct.byref(info)) == host.w.WGPUStatus_Success
        try:
            assert info.backendType == getattr(host.w, f"WGPUBackendType_{host.backend}")
            adapter = {
                "backend": host.backend,
                "software": info.adapterType == host.w.WGPUAdapterType_CPU,
                "description": host._message(info.description),
            }
        finally:
            host.w.wgpuAdapterInfoFreeMembers(info)
        shape = (1, 1, 2, 3, 4)
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["x", "x"], ["y"])],
            "native_bundle_arithmetic",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        )
        model = onnx.helper.make_model(
            graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 18)]
        )
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        buffers = [host.buffer(values.nbytes) for _ in range(2)]
        for buffer in buffers:
            resources.callback(buffer.close)
        host.write(buffers[0], values)
        session = GpuSession(host, model.SerializeToString())
        resources.callback(session.close)
        session.bind(
            {"x": DeviceTensor(buffers[0], shape, np.float32)},
            outputs={"y": DeviceTensor(buffers[1], shape, np.float32)},
        )
        assert "WebGpuExecutionProvider" in session.native_session.get_providers()
        session.submit()
        np.testing.assert_array_equal(host.read(buffers[1], np.float32).reshape(shape), values * 2)
        host.raise_errors()
        return {
            "status": "passed",
            "adapter": adapter,
            "providers": session.native_session.get_providers(),
            "shared_buffers": True,
            "physical_gpu_acceptance": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="native-bundle-") as temporary:
        assets = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(assets)
        manifest = json.loads((assets / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == digest, name
        assert ort.__version__ == manifest["onnxruntime"]
        report = {**check_provider(assets), "manifest": manifest}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in report.items() if key != "manifest"}, indent=2)
        )


if __name__ == "__main__":
    main()
