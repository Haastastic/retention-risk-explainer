# Deployment — Streamlit Community Cloud

Phase 8. Everything in the repo is ready for a free-tier deploy; the only step
that can't be automated is the one-time connect-and-deploy in the Streamlit
dashboard. This file is that step.

**Deployed:** <https://retention-risk-explainer.streamlit.app/> (branch `main`,
`ANTHROPIC_API_KEY` set → live Claude narration). It auto-redeploys on every push
to `main`. The steps below are kept as the runbook for re-deploying or moving it.

---

## What's already prepared

| File | Purpose |
|---|---|
| `app.py` | Streamlit entrypoint (repo root — the default Streamlit Cloud looks for) |
| `requirements.txt` | Runtime deps, version-bounded; installs clean on Python 3.12 |
| `runtime.txt` | Pins the Python version for Streamlit Cloud |
| `.streamlit/config.toml` | Theme + headless server config |
| `.streamlit/secrets.toml.example` | Template for the one optional secret |
| `data/raw/ibm_hr_attrition.csv` | Committed (~230 KB), so no runtime download |

The app **trains the model on cold start** (~4 s, cached for the session) rather
than loading a committed binary artifact — nothing to build or upload.

## One-time deploy (≈ 2 minutes)

1. Go to **<https://share.streamlit.io>** and sign in with the GitHub account
   that owns this repo (`Haastastic`).
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `Haastastic/retention-risk-explainer`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** your choice (e.g. `retention-risk-explainer`)
4. *(Optional — live Claude narration)* Expand **Advanced settings → Secrets**
   and paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Leave it blank to ship with the deterministic template narrator. The app
   works either way; the sidebar shows which mode is active.
5. **Deploy.** First build takes a few minutes (it installs `xgboost`, `shap`,
   `streamlit`). Subsequent boots are fast.

## After deploy

- The app **auto-redeploys on every push to `main`** — no further action.
- Put the live URL in the repo **About** panel and at the top of `README.md`.
- If the build fails on a dependency, check the build log in the Streamlit
  dashboard; the usual fix is a tighter version bound in `requirements.txt`.

## Verifying the deploy matches local

`tests/test_app.py::test_app_renders_without_exception` runs the whole page
headless in CI, so a green `main` means the app imports and renders. If the
deployed app errors but CI is green, it's almost always a dependency-resolution
difference — pin the offending package.

## Not using Streamlit Cloud?

The app is a standard Streamlit process:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Any host that can run that command (a container, a VM, Hugging Face Spaces) works
with no code change.
