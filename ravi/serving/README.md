# `ravi/serving`: where the endpoint lives

The Stable Audio 3 (rAvI LoRA) SageMaker endpoint is deployed from uapkg, next to the other NGMC
inference packages:

    ~/Development/uapkg/packages/system/algo/Sagemaker_Deployment/stable-audio-3-finetuned/
        code/inference.py        SageMaker hooks (model_fn / input_fn / predict_fn / output_fn)
        code/serving_core.py     request validation, mask planning, splice (pure numpy)
        code/ravi_vendor/        verbatim copies of ravi/eval/masks.py, ravi/structure/labels.py
                                 and parts of ravi/eval/sa3_infer.py (vendor_ravi.py writes them)
        parameter_schema.json    canvas form (schema_version 2, ngmc_poc lambda/types.ts)
        build_artifact.sh        model.tar.gz = code + adapters (+ optional base weights)
        DEPLOY.md                image, deploy, register and smoke-test commands; DLC vs BYOC
        smoke/validate_sa3_endpoint.py

The serving image is built from this repo: [`docker/sa3-serve`](../../docker/sa3-serve/)
(FROM `ravi-sa3-train`, plus a FastAPI `/ping` + `/invocations` server).

The handler reuses this package's inference logic instead of re-implementing it. The functions
are `Engine`, `choose_context`, `effective_latent_region`, `mask_seconds` and `to_model_audio`
from `ravi.eval.sa3_infer`, and `splice` from `ravi.eval.masks`. They are copied verbatim into
`code/ravi_vendor/`, because the tarball cannot import `ravi`.

**After changing `ravi/eval/masks.py`, `ravi/structure/labels.py` or `ravi/eval/sa3_infer.py`,
re-vendor:**

```bash
cd ~/Development/uapkg/packages/system/algo/Sagemaker_Deployment/stable-audio-3-finetuned
~/Development/brain-paper/.venv/bin/python vendor_ravi.py --brain-paper ~/Development/brain-paper
```

`ravi/tests/test_serving_contract.py` fails on drift. It also tests the request contract, the
mask padding/merging and the bit-identity of the splice, and runs `predict_fn` end to end with a
fake SA3 model. Point `RAVI_SA3_SERVING_CODE` at the handler's `code/` directory if uapkg is
elsewhere. The module skips itself when the handler is absent.
