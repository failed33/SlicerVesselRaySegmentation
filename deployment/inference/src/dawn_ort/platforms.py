"""Runtime library names and GPU backends.

These select assets and a backend; they do not establish platform acceptance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformConfig:
    dawn_library: str
    provider_library: str
    backend: str


_PLATFORMS = {
    "darwin": ("Metal", ".dylib"),
    "linux": ("Vulkan", ".so"),
    "win32": ("D3D12", ".dll"),
}


def platform_config(system: str) -> PlatformConfig:
    try:
        backend, suffix = _PLATFORMS[system]
    except KeyError:
        raise RuntimeError(f"No inference runtime configuration for {system}") from None
    prefix = "" if system == "win32" else "lib"
    return PlatformConfig(
        dawn_library=f"{prefix}webgpu_dawn{suffix}",
        provider_library=f"{prefix}onnxruntime_providers_webgpu{suffix}",
        backend=backend,
    )
