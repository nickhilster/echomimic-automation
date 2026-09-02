# echomimic-automation

Automating [EchoMimicV3](https://github.com/antgroup/echomimic_v3) (Ant Group's
audio-driven talking-portrait model, OSS) to take one reference image + one
audio clip and produce a lip-synced, body-animated video — running locally on
a Windows PC with a 12GB+ NVIDIA GPU.

## Status: environment setup scripted, inference wrapper not yet written

See [PLAN.md](PLAN.md) for the full writeup, and [TODO.md](TODO.md) for the
concrete next step.

## Quick start (on the GPU machine)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\echomimic_setup.ps1
```

Clones `antgroup/echomimic_v3`, creates a `echomimic_v3` conda env
(Python 3.10, CUDA 12.1 PyTorch), installs the repo's requirements, and
downloads the model weights from Hugging Face. Requires conda, git, and
ffmpeg already available on that machine.

## Repo layout

- `PLAN.md` — why EchoMimicV3/Flash was chosen over V1/V2, known Windows
  install gotchas, honesty notes on what's verified vs. not.
- `TODO.md` — the one open item blocking full automation.
- `scripts/echomimic_setup.ps1` — one-shot environment + weights setup.
