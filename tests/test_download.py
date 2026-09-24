import asyncio
import importlib.util
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import aiohttp


SPEC = importlib.util.spec_from_file_location(
    "booru_download_utils", pathlib.Path(__file__).resolve().parents[1] / "utils.py"
)
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


class Response:
    def __init__(self, status, data, headers, fail=False):
        self.status = status
        self.data = data
        self.headers = headers
        self.fail = fail
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(None, (), status=self.status)

    async def iter_chunked(self, _):
        yield self.data
        if self.fail:
            raise aiohttp.ClientPayloadError("connection closed early")


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append(kwargs["headers"])
        return next(self.responses)


async def no_sleep(_):
    pass


class DownloadTests(unittest.TestCase):
    def test_interrupted_transfer_resumes_and_publishes_complete_file(self):
        session = Session([
            Response(200, b"abc", {"Content-Length": "6", "ETag": '"v1"'}, fail=True),
            Response(206, b"def", {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'}),
        ])
        with tempfile.TemporaryDirectory() as directory, patch.object(utils.asyncio, "sleep", no_sleep):
            destination = pathlib.Path(directory) / "model.onnx_data"
            asyncio.run(utils.download_to_file("https://example.test/model", str(destination), session))
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertFalse(pathlib.Path(str(destination) + ".part").exists())
            self.assertEqual(session.requests[1]["Range"], "bytes=3-")
            self.assertEqual(session.requests[1]["If-Range"], '"v1"')

    def test_partial_file_survives_failed_call_and_resumes_next_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(utils.asyncio, "sleep", no_sleep):
            destination = pathlib.Path(directory) / "model.onnx_data"
            interrupted = Session([
                Response(200, b"a", {"Content-Length": "6", "ETag": '"v1"'}, fail=True),
                *[Response(206, b"", {"Content-Range": "bytes 1-5/6"}, fail=True)
                  for _ in range(4)],
            ])
            with self.assertRaises(aiohttp.ClientPayloadError):
                asyncio.run(utils.download_to_file("https://example.test/model", str(destination), interrupted))
            self.assertEqual(pathlib.Path(str(destination) + ".part").read_bytes(), b"a")
            resumed = Session([Response(206, b"bcdef", {"Content-Range": "bytes 1-5/6"})])
            asyncio.run(utils.download_to_file("https://example.test/model", str(destination), resumed))
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertEqual(resumed.requests[0]["Range"], "bytes=1-")

    def test_ignored_range_restarts_without_mixing_bytes(self):
        session = Session([
            Response(200, b"abc", {"Content-Length": "6"}, fail=True),
            Response(200, b"uvwxyz", {"Content-Length": "6"}),
        ])
        with tempfile.TemporaryDirectory() as directory, patch.object(utils.asyncio, "sleep", no_sleep):
            destination = pathlib.Path(directory) / "model.onnx_data"
            asyncio.run(utils.download_to_file("https://example.test/model", str(destination), session))
            self.assertEqual(destination.read_bytes(), b"uvwxyz")

    def test_completed_partial_file_is_published_after_range_416(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory) / "model.onnx_data"
            pathlib.Path(str(destination) + ".part").write_bytes(b"abcdef")
            pathlib.Path(str(destination) + ".part.json").write_text(
                '{"url": "https://example.test/model", "validator": "\\"v1\\""}')
            session = Session([Response(416, b"", {"Content-Range": "bytes */6"})])
            asyncio.run(utils.download_to_file("https://example.test/model", str(destination), session))
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertEqual(session.requests[0]["Range"], "bytes=6-")


if __name__ == "__main__":
    unittest.main()
