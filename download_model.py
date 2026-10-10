"""Download public weights at a fixed revision; verify before local use."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent / "model"
REVISION = "4b544e74dfc3256b2b56849ea5d7064fee1ac846"
BASE = f"https://huggingface.co/Xenova/paraphrase-MiniLM-L3-v2/resolve/{REVISION}"
FILES = {
    "model_quantized.onnx": ("onnx/model_quantized.onnx", "b190f50dd46296b9895ae8f274c3455762d08610f8788f4a9bd15019f4f7160c"),
    "tokenizer.json": ("tokenizer.json", "2fc687b11de0bc1b3d8348f92e3b49ef1089a621506c7661fbf3248fcd54947e"),
}

if __name__ == "__main__":
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,(remote,expected) in FILES.items():
        path = ROOT/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            with urllib.request.urlopen(f"{BASE}/{remote}",timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest()!=expected:
                raise RuntimeError(f"Hash mismatch for {name}; file was not saved")
            path.write_bytes(payload)
        print(f"Verified {name}: {path.stat().st_size:,} bytes")
