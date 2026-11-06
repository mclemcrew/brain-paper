"""SageMaker bring-your-own-container server for the Stable Audio 3 (rAvI LoRA) endpoint.

SageMaker starts the container as ``docker run <image> serve`` and talks to port 8080:
``GET /ping`` (200 once the model is loaded) and ``POST /invocations`` (the request body;
for an async endpoint SageMaker reads it from the S3 InputLocation and writes our response
body to the S3 output location). The model code is NOT in this image: it is the uapkg
handler ``Sagemaker_Deployment/stable-audio-3-finetuned/code/inference.py``, which arrives
in ``model.tar.gz`` (extracted to ``/opt/ml/model``). This server only imports it and calls
its four SageMaker hooks, so the DLC and BYOC paths run the same handler.

Env: ``SM_MODEL_DIR`` (default /opt/ml/model), ``SA3_HANDLER_DIR`` (default
$SM_MODEL_DIR/code), ``SAGEMAKER_BIND_TO_PORT`` (8080), ``SA3_EXIT_ON_LOAD_FAILURE``
(default 1: exit so the deployment fails fast with the traceback in CloudWatch instead of
waiting out the health-check timeout).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
HANDLER_DIR = os.environ.get("SA3_HANDLER_DIR", os.path.join(MODEL_DIR, "code"))
PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
EXIT_ON_LOAD_FAILURE = os.environ.get("SA3_EXIT_ON_LOAD_FAILURE", "1") == "1"

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [sa3-serve] %(levelname)s %(message)s")
log = logging.getLogger("sa3-serve")

STATE: dict = {"handler": None, "model": None, "error": None, "fatal": None, "started": time.time(),
               "ready_after_s": None}
_LOCK = threading.Lock()  # one GPU, one request at a time (the handler also locks)


def load_handler():
    path = os.path.join(HANDLER_DIR, "inference.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found: is model.tar.gz built with build_artifact.sh?")
    if HANDLER_DIR not in sys.path:
        sys.path.insert(0, HANDLER_DIR)
    spec = importlib.util.spec_from_file_location("inference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inference"] = mod
    spec.loader.exec_module(mod)
    for fn in ("model_fn", "input_fn", "predict_fn", "output_fn"):
        if not callable(getattr(mod, fn, None)):
            raise AttributeError(f"{path} does not define {fn}")
    return mod


def _load() -> None:
    try:
        handler = load_handler()
        log.info("loading model from %s with %s", MODEL_DIR, handler.__file__)
        model = handler.model_fn(MODEL_DIR)
        STATE.update(handler=handler, model=model, ready_after_s=round(time.time() - STATE["started"], 1))
        log.info("ready after %.1f s", STATE["ready_after_s"])
    except BaseException as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        log.error("model load failed:\n%s", traceback.format_exc())
        if EXIT_ON_LOAD_FAILURE:
            logging.shutdown()
            os._exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load, name="model-load", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/ping")
def ping():
    if STATE["fatal"]:  # e.g. a sticky CUDA error: unhealthy so SageMaker replaces the container
        return JSONResponse({"status": "error", "error": STATE["fatal"]}, status_code=500)
    if STATE["model"] is not None:
        return JSONResponse({"status": "ok", "ready_after_s": STATE["ready_after_s"]})
    if STATE["error"]:
        return JSONResponse({"status": "error", "error": STATE["error"]}, status_code=500)
    return JSONResponse({"status": "loading"}, status_code=503)


def _invoke(body: bytes, content_type: str, accept: str):
    h, model = STATE["handler"], STATE["model"]
    with _LOCK:
        data = h.input_fn(body, content_type)
        pred = h.predict_fn(data, model)
        out = h.output_fn(pred, accept)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, accept


@app.post("/invocations")
async def invocations(request: Request):
    if STATE["model"] is None:
        return JSONResponse({"error": STATE["error"] or "model is still loading"}, status_code=503)
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")
    accept = request.headers.get("accept", "application/json")
    t0 = time.time()
    try:
        payload, media_type = await run_in_threadpool(_invoke, body, content_type, accept)
    except Exception as e:  # noqa: BLE001
        # only the handler's own RequestError is a client error; any other ValueError is a bug
        request_error = getattr(getattr(STATE["handler"], "core", None), "RequestError", None)
        if request_error is not None and isinstance(e, request_error):
            log.warning("400 after %.1f s: %s", time.time() - t0, e)
            return JSONResponse({"error": str(e)}, status_code=400)
        log.error("500 after %.1f s:\n%s", time.time() - t0, traceback.format_exc())
        if "CUDA" in str(e) or "cuda" in type(e).__name__.lower():
            STATE["fatal"] = f"{type(e).__name__}: {e}"
            log.error("CUDA error: the GPU context is unusable; /ping now fails and the process exits")
            threading.Timer(2.0, lambda: os._exit(1)).start()  # let this 500 go out first
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    return Response(content=payload, media_type=media_type)


def main() -> None:
    # `docker run <image> serve` passes "serve"; there is nothing else to dispatch on.
    if len(sys.argv) > 1 and sys.argv[1] not in ("serve",):
        log.warning("ignoring arguments %s", sys.argv[1:])
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1, timeout_keep_alive=75, log_level="info")


if __name__ == "__main__":
    main()
