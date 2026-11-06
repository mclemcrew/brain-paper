# `sagemaker-stable-audio-3-finetuned`: Stable Audio 3 inference image (SageMaker BYOC)

The serving image for the `stable-audio-3-finetuned-dev` endpoint. It is built `FROM` the
`ravi-sa3-train` image (Python 3.10, torch 2.7.1+cu126, flash-attn 2.6.3, stable-audio-3 at
77e0398c, all from SA3's `uv.lock`), plus FastAPI 0.115.12 and uvicorn 0.34.0.

`server.py` answers SageMaker's container contract: `docker run <image> serve`, port 8080,
`GET /ping`, `POST /invocations`. `/ping` returns 503 while loading and 200 once loaded. If the
load fails, the container exits so the deployment fails fast. The server holds no model code. It
imports `/opt/ml/model/code/inference.py` from the model artifact, i.e. the uapkg handler
`Sagemaker_Deployment/stable-audio-3-finetuned/code/`, and calls its `model_fn / input_fn /
predict_fn / output_fn`. A `ValueError` from the handler becomes HTTP 400; anything else becomes
500. Requests are serialised: one GPU, one request at a time.

Why not the stock `pytorch-inference:2.6.0` DLC: SA3 pins torch 2.7.1 and needs flash-attn. The
evidence is in the uapkg package's `DEPLOY.md`, "Why BYOC".

```bash
PUSH=0 PROFILE= docker/sa3-serve/build_and_push.sh          # on the GPU box: local build from ravi-sa3-train:dev
docker/sa3-serve/build_and_push.sh                          # push to ECR sagemaker-stable-audio-3-finetuned:<sha>-<UTC time>
BASE_IMAGE=<ecr>/ravi-sa3-train:<tag> docker/sa3-serve/build_and_push.sh
```

Env: `SM_MODEL_DIR` (default `/opt/ml/model`), `SA3_HANDLER_DIR` (`$SM_MODEL_DIR/code`),
`SAGEMAKER_BIND_TO_PORT` (8080), `SA3_EXIT_ON_LOAD_FAILURE` (1). For local tests only,
`SA3_ALLOW_FILE_SOURCES=1` lets requests use `file:///…` sources instead of S3.

Verified 2026-10-15 on the GPU box (L40S): 22/22 HTTP checks. They cover generate determinism,
adapter strength 0 = base bit-for-bit, an adapter detach/attach cycle that restores the base
exactly, and inpaint bit-identical outside the mask for WAV and 48 kHz MP3 sources, with two
masks. They also cover continue (mid-track and append) and 400 on bad requests. The model loads
in 26 s from local files and uses ~10.3 GB of GPU memory.
