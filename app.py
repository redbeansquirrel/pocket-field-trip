"""Pocket Field Trip: actual local MiniLM inference, no remote fallback."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import argparse, hashlib, json, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MISSIONS = json.loads((ROOT / "missions.json").read_text(encoding="utf-8"))
MODEL = ROOT / "model" / "model_quantized.onnx"
LOCK = threading.Lock()

class Planner:
    def __init__(self):
        started = time.perf_counter()
        self.tokenizer = Tokenizer.from_file(str(ROOT / "model" / "tokenizer.json"))
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=128)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])
        self.embeddings = self.encode([m["description"] for m in MISSIONS])
        self.startup_ms = round((time.perf_counter() - started) * 1000, 1)

    def encode(self, texts):
        batch = self.tokenizer.encode_batch(texts)
        tensors = {
            "input_ids": np.array([x.ids for x in batch], dtype=np.int64),
            "attention_mask": np.array([x.attention_mask for x in batch], dtype=np.int64),
            "token_type_ids": np.array([x.type_ids for x in batch], dtype=np.int64),
        }
        feed = {x.name: tensors[x.name] for x in self.session.get_inputs()}
        raw = self.session.run(None, feed)[0]
        mask = tensors["attention_mask"][..., None]
        pooled = (raw * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    def plan(self, request):
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object.")
        query = request.get("query", "")
        if not isinstance(query, str):
            raise ValueError("Please describe your break as text.")
        query = query.strip()
        if not query or len(query) > 400:
            raise ValueError("Please describe your break in 1 to 400 characters.")
        minutes = request.get("minutes", 10)
        if type(minutes) is not int:
            raise ValueError("Time available must be a whole number of minutes.")
        if minutes < 5 or minutes > 60:
            raise ValueError("Choose between 5 and 60 minutes.")
        walking = request.get("walking", True)
        outdoor_ok = request.get("outdoor_ok", False)
        if type(walking) is not bool or type(outdoor_ok) is not bool:
            raise ValueError("Movement and outdoor conditions must be true or false.")
        if not outdoor_ok:
            return {"status":"paused", "message":"Choose another time when your nearby outdoor space and conditions are suitable."}
        candidates = [(i,m) for i,m in enumerate(MISSIONS) if m["minutes"] <= minutes and (walking or not m["walk"])]
        if not candidates:
            return {"status":"no_match", "message":"No suitable mission in this small collection. Try a longer break."}
        started = time.perf_counter()
        with LOCK:
            query_vector = self.encode([query])[0]
        scored = sorted([(float(self.embeddings[i] @ query_vector),m) for i,m in candidates], key=lambda x:x[0], reverse=True)
        score, mission = scored[0]
        return {"status":"ok", "mission":mission, "similarity":round(score, 4),
                "inference_ms":round((time.perf_counter()-started)*1000,1),
                "model":"paraphrase-MiniLM-L3-v2 / quantized ONNX / CPU",
                "alternatives":[{"id":m["id"],"similarity":round(s,4)} for s,m in scored[1:3]],
                "note":"Similarity ranks this small collection; it is not a confidence or suitability score."}

def evaluate(planner, output):
    # For this mode, any Python socket connection attempt is an error.
    # The model was already loaded entirely from explicit local paths.
    def deny(*args, **kwargs):
        raise RuntimeError("Network forbidden during offline evaluation")
    socket.create_connection = deny
    socket.socket.connect = deny
    cases = [
        ("I want to get some exercise and move my legs", "brisk-loop"),
        ("My kids need a playful hunt for colorful things", "color-hunt"),
        ("I want to draw something with my pencil", "shadow-sketch"),
        ("I want to listen carefully to the sounds around me", "sound-map"),
        ("I would like to observe what birds do", "bird-watch"),
        ("I want to study the details of a plant leaf", "leaf-study"),
        ("I'd like a conversation with a friend outdoors", "friend-loop"),
        ("I want to sit and see the shapes of clouds", "cloud-watch"),
    ]
    results = []
    for query, expected in cases:
        result = planner.plan({"query":query,"minutes":20,"walking":True,"outdoor_ok":True})
        results.append({"query":query,"expected":expected,"actual":result["mission"]["id"],"passed":expected==result["mission"]["id"],"result":result})
    constraints = []
    for minutes in [5,7,10,15]:
        for walking in [True,False]:
            for query,_ in cases:
                r = planner.plan({"query":query,"minutes":minutes,"walking":walking,"outdoor_ok":True})
                constraints.append(r["status"] != "ok" or (r["mission"]["minutes"]<=minutes and (walking or not r["mission"]["walk"])))
    paused = planner.plan({"query":"go hiking","minutes":20,"outdoor_ok":False})
    report = {"created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"stage":"local prototype, unpublished, no earnings",
        "model_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),"model_bytes":MODEL.stat().st_size,
        "tokenizer_bytes":(ROOT/"model"/"tokenizer.json").stat().st_size,
        "runtime":ort.__version__,"providers":planner.session.get_providers(),"embedding_shape":list(planner.embeddings.shape),
        "finite_embeddings":bool(np.isfinite(planner.embeddings).all()),"startup_ms":planner.startup_ms,
        "network_guard":"Python sockets denied during all evaluation requests; no remote inference code exists",
        "semantic_cases":{"passed":sum(x["passed"] for x in results),"total":len(results),"cases":results},
        "constraints":{"passed":sum(constraints),"total":len(constraints)},"outdoor_pause_passed":paused["status"]=="paused",
        "limitations":["Hand-authored diagnostic examples, not an independent benchmark.","English-only model and UI.","No outdoor field trial or behavioral benefit has been measured.","Curated mission text; model only ranks missions, it does not generate the instructions."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["semantic_cases","limitations"]},indent=2))
    print("Semantic cases:",report["semantic_cases"]["passed"],"/",len(results))

def serve(planner, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # Do not persist users' text.
        def do_GET(self):
            if self.path not in ["/","/index.html"]:
                self.send_error(404); return
            content = (ROOT/"index.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(content)
        def do_POST(self):
            if self.path != "/api/plan": self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length",0))
                if length < 1 or length > 4096: raise ValueError("Invalid request length")
                result = planner.plan(json.loads(self.rfile.read(length)))
                body = json.dumps(result).encode(); self.send_response(200)
            except (ValueError,TypeError,KeyError) as exc:
                body = json.dumps({"status":"error","message":str(exc)}).encode(); self.send_response(400)
            self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(body)
    print(f"Pocket Field Trip is ready at http://127.0.0.1:{port}; model startup {planner.startup_ms} ms",flush=True)
    ThreadingHTTPServer(("127.0.0.1",port),Handler).serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate",type=Path)
    parser.add_argument("--port",type=int,default=8766)
    args = parser.parse_args()
    planner = Planner()
    if args.evaluate: evaluate(planner,args.evaluate)
    else: serve(planner,args.port)
