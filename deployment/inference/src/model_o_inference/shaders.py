"""Load generated Slang resources without a runtime compiler dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from .constants import SHADER_CONSTANTS

_BUFFER = TypeVar("_BUFFER")
_ROOT = Path(__file__).with_name("webgpu")


@dataclass(frozen=True)
class Shader:
    code: str
    metadata: dict[str, Any]

    @property
    def entry_point(self) -> str:
        return self.metadata["entryPoints"][0]["name"]

    def workgroups(self, elements: int) -> int:
        size, y, z = self.metadata["entryPoints"][0]["threadGroupSize"]
        if y != 1 or z != 1:
            raise ValueError("ModelO dispatch requires a one-dimensional workgroup")
        return (elements + size - 1) // size

    def bindings(self, buffers: dict[str, _BUFFER]) -> dict[int, _BUFFER]:
        parameters = self.metadata["parameters"]
        if set(buffers) != {p["name"] for p in parameters}:
            raise ValueError("Shader resources do not match supplied buffers")
        bindings = {}
        for parameter in parameters:
            binding = parameter["binding"]
            resource = parameter["type"]
            if (
                binding["kind"] != "descriptorTableSlot"
                or binding.get("space", 0) != 0
                or resource["kind"] != "resource"
                or resource["baseShape"] != "structuredBuffer"
            ):
                raise ValueError("Only group-zero structured storage buffers are supported")
            if binding["index"] in bindings:
                raise ValueError("Duplicate shader binding")
            bindings[binding["index"]] = buffers[parameter["name"]]
        return bindings


@cache
def load_shader(name: str) -> Shader:
    if name not in {"patch_inputs", "sigmoid"}:
        raise ValueError(f"Unknown ModelO shader: {name}")
    code = (_ROOT / "generated" / f"{name}.wgsl").read_bytes()
    metadata = json.loads((_ROOT / "generated" / f"{name}.json").read_text())
    build = metadata["build"]
    if hashlib.sha256(code).hexdigest() != build["wgsl_sha256"]:
        raise ValueError(f"Generated WGSL digest mismatch: {name}")
    if hashlib.sha256((_ROOT / f"{name}.slang").read_bytes()).hexdigest() != build["source_sha256"]:
        raise ValueError(f"Slang source digest mismatch: {name}")
    if build["constants"] != SHADER_CONSTANTS or SHADER_CONSTANTS["RAY_LENGTH"] % 4:
        raise ValueError("Generated shader and ModelO geometry disagree")
    if len(metadata["entryPoints"]) != 1 or metadata["entryPoints"][0]["stage"] != "compute":
        raise ValueError("Expected one compute entry point")
    return Shader(code.decode(), metadata)
