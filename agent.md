# Agent Notes

## Colab Interaction

- Colab pages loaded from GitHub usually show a safety confirmation before the first execution. Always look for the warning and click **Run anyway** before assuming a cell did not run.
- Colab output panes can hide errors or render stale output after repeated edits. Prefer short, verifiable cells when debugging, and print explicit markers such as `START`, `HEAD`, and `END`.
- Long pasted cells can be visually wrapped by Colab/Monaco. The visible wrapping is usually harmless, but if execution behaves oddly, replace the cell with smaller steps rather than continuing to paste larger scripts.
- Colab runtimes reset often. Before using `--skip-download` or assuming files exist, check paths such as `/content/FloodRoad-SAM3`, `/content/spacenet8/raw`, and `/content/floodroad_runs/default`.
- GitHub-hosted notebooks may need a one-time safety prompt every reload. If a run appears to do nothing, re-check for the modal before rerunning.

## GitHub Token In Colab

- For private repos, do not rely on plain `git clone https://github.com/...` inside Colab. It can block at `Username for 'https://github.com':` because Colab has no GitHub credentials.
- Store a GitHub PAT in Colab Secrets as `github`. The token needs read access to `ezh2020/FloodRoad-SAM3`; for fine-grained PATs, grant repository access and `Contents: Read-only`.
- Never print `userdata.get('github')`. If a token is printed in notebook output, rotate/revoke it in GitHub immediately.
- Use non-interactive clone commands and mask the token in captured output:

```python
from google.colab import userdata
import os
import shutil
import subprocess

github_token = userdata.get("github")
assert github_token, "Missing Colab Secret named github"
repo_url = f"https://{github_token}@github.com/ezh2020/FloodRoad-SAM3.git"

shutil.rmtree("/content/FloodRoad-SAM3", ignore_errors=True)
env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
result = subprocess.run(
    ["git", "clone", repo_url, "/content/FloodRoad-SAM3"],
    text=True,
    capture_output=True,
    env=env,
)
print(result.stdout.replace(github_token, "***"))
print(result.stderr.replace(github_token, "***"))
assert result.returncode == 0, result.returncode

repo_url = None
github_token = None
```

- If clone fails with `Invalid username or token`, the secret is missing, expired, revoked, or lacks access to the target repo. Update the Colab Secret after rotating the PAT.

## Hugging Face Token In Colab

- Store the Hugging Face token as Colab Secret `HF_TOKEN` and never print it.
- Load it into both environment variable names used by Python packages:

```python
import os
from google.colab import userdata

if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
    hf_token = userdata.get("HF_TOKEN")
    assert hf_token, "Missing Colab Secret named HF_TOKEN"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
```

- SAM3 checkpoints are gated. The Hugging Face account behind `HF_TOKEN` must have accepted access to `facebook/sam3`.

## Dependency Notes

- Official SAM3 currently requires `numpy>=1.26,<2`. Keep Colab geospatial dependencies compatible with that constraint (`rasterio<1.5`, `opencv-python-headless<4.11`).
- Warnings from pip about unrelated Colab packages can be non-fatal, but import-test the actual stack before launching a long run:

```python
import numpy, rasterio, cv2, sam3
print("deps ok", numpy.__version__, rasterio.__version__, cv2.__version__)
```

## Experiment Run Notes

- The formal run should use the real SpaceNet 8 data path and not toy data.
- If `/content/spacenet8/raw` is missing, do not pass `--skip-download`; the runner must download the SpaceNet tarball again.
- If `/content/spacenet8/raw` already exists and only code changed, use `--skip-download` to reuse the downloaded tarball.
