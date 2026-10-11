"""Check runtime platform names independently of the native build recipe."""

import pytest
from dawn_ort.platforms import platform_config


@pytest.mark.parametrize(
    ("system", "backend", "suffix"),
    [("darwin", "Metal", ".dylib"), ("linux", "Vulkan", ".so"), ("win32", "D3D12", ".dll")],
)
def test_platform_libraries_agree(system, backend, suffix):
    config = platform_config(system)
    assert config.backend == backend
    assert all(name.endswith(suffix) for name in [config.dawn_library, config.provider_library])


def test_unconfigured_runtime_platform_is_rejected():
    with pytest.raises(RuntimeError, match="No inference runtime configuration"):
        platform_config("freebsd")
