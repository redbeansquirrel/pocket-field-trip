# Pocket Field Trip

A local AI tool that turns an English description of the break you want into one short outdoor mission. Read the card, put the screen away, and go. Built from scratch on October 10, 2026 for the Hacktoberfest Week 1 theme, Touch Grass.

**Status:** source and demonstration prepared for public release; challenge entry pending. No award or earnings. An outdoor field trial has not been performed.

## What actually runs

The quantized `paraphrase-MiniLM-L3-v2` model computes 384-dimensional embeddings on the local CPU using ONNX Runtime. The model ranks eight original, curated missions by semantic similarity. Deterministic checks filter by time and whether walking is welcome. An unsuitable-outdoor-conditions answer pauses the activity.

The instructions on the selected card are authored text. They are not claimed to be model-generated. There are no inference APIs, API keys, telemetry, account logins, location requests, or background network calls in the app. The model is loaded only from explicit local paths. Raw requests are not logged or saved.

## Install and run

Use Python 3.12 on Windows. Extract the source archive, open PowerShell in the extracted folder, and run:

```powershell
py -3.12 -m pip install --target vendor -r requirements-lock.txt
py -3.12 download_model.py
py -3.12 app.py --port 8766
```

If the Windows Python launcher is unavailable, replace `py -3.12` with your Python 3.12 executable command. Open <http://127.0.0.1:8766>. The server binds only to the computer's loopback interface. If another copy is already running, use its existing page or choose another port.

Only initial installation/download requires the internet. Public weights plus tokenizer total **18,163,755 bytes**, about 18.2 MB (17.3 MiB). The installed dependency directory on the tested machine totals **211,267,942 bytes**. Including the model, allow about **230 MB** of disk space. Package downloads total roughly 43 MB plus the model; exact transfer depends on cache and platform. Initial setup took under three minutes on this connection; budget 2–5 minutes on a normal connection and longer on a slow one. No GPU is needed.

The downloader pins upstream revision `4b544e74dfc3256b2b56849ea5d7064fee1ac846` and verifies both SHA-256 values. The model and dependencies are excluded from the source repository; the app fails visibly if they are missing. It does not silently replace the model with mock outputs.

## Evaluation

```powershell
py -3.12 app.py --evaluate evaluation.json
```

The initial local run produced 8/8 expected matches on hand-written semantic diagnostic examples, 64/64 time-and-movement constraint checks, and a successful unsuitable-conditions pause check. Python socket connections were disabled during the evaluation requests. Model startup was 80.5 ms; each of the eight individual example requests took 0.6–1.3 ms on the test computer after startup. These are local observations, not hardware-independent guarantees. The command above generates an `evaluation.json` file with fresh per-request values and the model hash on your computer.

An additional HTTP boundary check passed 29 malformed input cases, each returning a JSON HTTP 400 response, and three valid controls covering an exercise request, a no-walking constraint and the outdoor pause. After the input-validation fix, the original semantic and constraint checks still passed. Reproduce the HTTP check with:

```powershell
py -3.12 test_invalid_inputs.py
```

The examples were written by the builder alongside the app. They are not an independent benchmark, and do not establish real-world recommendation accuracy or health benefits. The app is English only, knows no local weather or route conditions, and performs no species identification. Similarity is a ranking signal, not a calibrated confidence score. The small mission collection is visible and editable in `missions.json`.

Browser review also exposed a relevance limit: asking for exercise while disabling walking can return a drawing activity. The movement restriction is honored, but the remaining card can be a weak match for the original request. The eight-mission collection does not guarantee a useful match for every combination of interests and constraints.

## Demo

A 60-second demonstration has been generated from five actual browser screenshots, each shown for 12 seconds with explanatory captions. It is an edited screenshot demonstration, not a continuous screen recording. The browser review covers:

1. The initial interface.
2. A drawing request producing the shadow-sketch card.
3. An exercise request producing the familiar brisk loop.
4. The no-walking constraint, including the relevance limitation described above.
5. The pause message when outdoor conditions are unsuitable.

Watch the [60-second demo](https://redbeansquirrel.github.io/pocket-field-trip/) or [download the video](docs/demo.mp4). The website hosts the demonstration; the AI application runs locally. The [recorded post-fix evaluation](docs/evaluation.json) is also included. Challenge submission status is separate from this source publication.

## Attribution

- Original model: [Sentence Transformers MiniLM-L3 model card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2), Apache-2.0.
- Quantized ONNX conversion: [Xenova model repository](https://huggingface.co/Xenova/paraphrase-MiniLM-L3-v2), fixed revision recorded above.
- Inference: [ONNX Runtime](https://onnxruntime.ai/), MIT.
- Tokenization: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers), Apache-2.0.

AI assisted implementation, original mission writing and documentation. Source code in this folder is offered under the attached MIT license. Upstream components retain their own licenses.
