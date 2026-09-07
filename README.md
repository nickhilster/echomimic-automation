# echomimic-automation

Automating [EchoMimicV3](https://github.com/antgroup/echomimic_v3) (Ant Group's
audio-driven talking-portrait model) to take one reference image + one audio
clip and produce a lip-synced, body-animated video — running locally on a
Windows PC with a 12 GB NVIDIA GPU.

## Status: working end-to-end on RTX 4070 (12 GB) + 16 GB RAM

demo 01/02 produce coherent lip-synced video (560x1024, 81 frames @ 25 fps,
8 steps, ~5-6 min/run). See [PLAN.md](PLAN.md) for the full writeup and
[TODO.md](TODO.md).

## Setup (one time, on the GPU machine)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\echomimic_setup.ps1
```

Clones `antgroup/echomimic_v3`, builds the `echomimic_v3` conda env (Python 3.10,
CUDA 12.1 PyTorch) with a **pinned** dependency set, installs requirements,
downloads ~24 GB of weights, converts the wav2vec weights to safetensors.
Needs conda + git + ffmpeg on PATH.

## Generate a video

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_video.ps1 -Image face.jpg -Audio speech.wav -Output out.mp4
```

Defaults to the low-RAM driver. `-Steps 20` for talking-body clips; see
[PLAN.md](PLAN.md) for all options.

Clips longer than one model window (~81 frames / ~3.2 s) are generated
automatically in overlapping chunks and cross-faded. Set total length with
`-VideoLength` (seconds x 25); tune `-PartialLength` / `-Overlap` if needed.
Verified: a 15 s clip = five 81-frame chunks, ~8 min on the 4070.

## Repo layout

- `PLAN.md` — status, the low-RAM driver rationale, every Windows/dependency
  gotcha and how the scripts handle it.
- `TODO.md` — what's done, what's optional next.
- `scripts/echomimic_setup.ps1` — one-shot env + pinned deps + weights + wav2vec conversion.
- `scripts/infer_lowram.py` — load-encode-free inference driver (fits ~16 GB RAM).
- `scripts/generate_video.ps1` — `-Image` + `-Audio` + `-Output` wrapper.
- `scripts/patch_infer_flash.py` — makes upstream `infer_flash.py` honour
  `--GPU_memory_mode` (only for `-Driver flash`).
