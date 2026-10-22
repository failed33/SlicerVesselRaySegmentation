"""Validate Slicer runtime download verification and retry behavior."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.runtimes.download_slicer import download


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _build_archive(executable_root: str) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as package:
        data = b"#!/bin/sh\nexit 0\n"
        entry = tarfile.TarInfo(name=f"{executable_root}/Slicer")
        entry.mode = 0o755
        entry.size = len(data)
        package.addfile(entry, io.BytesIO(data))
    return payload.getvalue()


def _metadata(archive_bytes: bytes, version: str, suffix: str) -> dict[str, object]:
    return {
        "download_url": f"/bitstream/{suffix}",
        "sha512": hashlib.sha512(archive_bytes).hexdigest(),
        "size": len(archive_bytes),
        "version": version,
    }


class DownloadSlicerTest(unittest.TestCase):
    def test_retries_when_checksum_verification_fails(self) -> None:
        bad_archive = b"not-a-valid-slicer-package"
        good_archive = _build_archive("Slicer-good")
        metadata_attempt_1 = _metadata(_build_archive("Slicer-other"), "5.13.0", "first")
        metadata_attempt_2 = _metadata(good_archive, "5.13.0", "second")
        responses = iter(
            [
                _Response(json.dumps(metadata_attempt_1).encode()),
                _Response(bad_archive),
                _Response(json.dumps(metadata_attempt_2).encode()),
                _Response(good_archive),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "tools.runtimes.download_slicer.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: next(responses),
            ):
                values = download("nightly", Path(temporary))
            self.assertTrue(Path(values["slicer_executable"]).exists())
            self.assertEqual(values["slicer_version"], "5.13")
            self.assertEqual(
                json.loads((Path(temporary) / "download.json").read_text()),
                metadata_attempt_2,
            )

    def test_fails_after_three_mismatched_attempts(self) -> None:
        metadata = _metadata(_build_archive("Slicer-expected"), "5.13.0", "same")
        responses = iter(
            [
                _Response(json.dumps(metadata).encode()),
                _Response(b"attempt-1"),
                _Response(json.dumps(metadata).encode()),
                _Response(b"attempt-2"),
                _Response(json.dumps(metadata).encode()),
                _Response(b"attempt-3"),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "tools.runtimes.download_slicer.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: next(responses),
            ):
                with self.assertRaisesRegex(
                    ValueError, "Slicer download size or SHA-512 verification failed"
                ):
                    download("nightly", Path(temporary))
            self.assertFalse((Path(temporary) / "slicer.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
