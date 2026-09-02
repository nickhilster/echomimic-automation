# EchoMimicV3 on your Windows GPU box — automation plan

## What this gets you
EchoMimicV3 (Flash variant) generates a talking-head/talking-body video from one
reference image + one audio clip, with lip sync and body motion driven by the audio.
It's the EchoMimic variant Ant Group actually built for ~12GB VRAM cards — V1/V2
were only validated down to 16GB.

## Step 1 — run the setup script (one time)
Copy `echomimic_setup.ps1` to the GPU machine and run:

```powershell
powershell -ExecutionPolicy Bypass -File echomimic_setup.ps1
```

It clones `antgroup/echomimic_v3`, creates a `echomimic_v3` conda env with
Python 3.10 + CUDA 12.1 PyTorch, installs the repo's `requirements.txt`, and
pulls the model weights from Hugging Face (several GB — the base
`Wan2.1-Fun-V1.1-1.3B-InP` DiT model plus `BadToBest/EchoMimicV3` weights).

Prerequisites it assumes are already on that machine: **conda** (Miniconda),
**git**, and **ffmpeg on PATH**. The script checks for these and tells you what's
missing.

## Step 2 — confirm the inference entrypoint (I can't verify this remotely)
This repo moves fast and its exact CLI/config shape wasn't independently
confirmed. After setup finishes, open the cloned repo
(`%USERPROFILE%\EchoMimicV3`) and check `README.md` for the current Flash
inference command — as of this research it was one of:

```
bash run_flash.sh
```
or
```
python app_mm.py     # Gradio web UI, quantized, fits 12GB
```

Both read image/audio paths from a config file (YAML/JSON under `configs/`)
rather than pure CLI flags — that's the pattern across all EchoMimic versions.

## Step 3 — send me the actual config file
Once you've run setup and can see the real config file (e.g.
`configs/prompts/*.yaml` or similar), paste its contents back to me — I'll
write you a one-command wrapper script (`generate_video.ps1 -Image face.jpg
-Audio speech.wav -Output out.mp4`) that patches the config and runs
inference for you, so from then on it's fully automated on your end.

## Known Windows friction points
- `xformers` / flash-attention have no official Windows wheels — if pip
  install fails on those, either skip them (V3 Flash doesn't strictly require
  xformers) or grab a prebuilt wheel matching your torch/CUDA version from
  https://github.com/wildminder/AI-windows-whl
- If native Windows install breaks on a compiled dependency, WSL2 (Ubuntu)
  + the same conda steps is the fallback most users report success with.
- Ignore the repo's Windows "one-click installer" — it's distributed only via
  Baidu Netdisk, which isn't something to trust/run from here.

## Honesty check on "fully automate"
I can script the entire environment setup and the run command shape reliably.
The one piece I can't do sight-unseen is guarantee the *exact* config keys
EchoMimicV3's current inference script expects, since I don't have a GPU here
to test against and the repo's docs on that specific point are thin. Step 3
closes that gap in one round trip once you have the repo cloned.
