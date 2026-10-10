"""Exercise the real HTTP boundary with malformed inputs and valid controls."""
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
VALID = {"query": "I want to get some exercise and move my legs",
         "minutes": 15, "walking": True, "outdoor_ok": True}


class InputBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        cls.server = subprocess.Popen(
            [sys.executable, "-B", str(ROOT / "app.py"), "--port", str(cls.port)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if cls.server.poll() is not None:
                raise RuntimeError(cls.server.stderr.read().decode(errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("Local test server did not become ready")

    @classmethod
    def stop_server(cls):
        if cls.server.poll() is None:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server.kill()
                cls.server.wait(timeout=5)
        cls.server.stderr.close()

    def post(self, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", "/api/plan", body=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_invalid_types_return_json_400(self):
        cases = [None, [], "hello", 12, True]
        for field, values in {
            "query": [None, 42, True, [], {}],
            "minutes": [None, True, False, "15", 15.5, [], {}],
            "walking": [None, "false", 0, 1, [], {}],
            "outdoor_ok": [None, "false", 0, 1, [], {}],
        }.items():
            cases.extend(dict(VALID, **{field: value}) for value in values)
        for payload in cases:
            with self.subTest(payload=payload):
                status, result = self.post(payload)
                self.assertEqual(status, 400)
                self.assertEqual(result["status"], "error")
                self.assertIsInstance(result["message"], str)

    def test_valid_requests_keep_semantics(self):
        status, result = self.post(VALID)
        self.assertEqual(status, 200)
        self.assertEqual(result["mission"]["id"], "brisk-loop")
        status, result = self.post(dict(VALID, walking=False))
        self.assertEqual(status, 200)
        self.assertFalse(result["mission"]["walk"])
        self.assertLessEqual(result["mission"]["minutes"], VALID["minutes"])
        status, result = self.post(dict(VALID, outdoor_ok=False))
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "paused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
