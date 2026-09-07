# EchoMimicV3 on a Windows 12 GB GPU box — status & notes

## What this does
EchoMimicV3 (Flash variant) turns one reference image + one audio clip into a
lip-synced, body-animated talking video. Ant Group built the Flash variant for
~12 GB VRAM cards.

## Verified working
- **Box:** Windows 11, RTX 4070 (12 GB VRAM), 16 GB system RAM.
- **First result:** demo 01 -> 560x1024, 81 frames @ 25 fps, 8 steps,
  ~3 min of diffusion + ~2 min model load, coherent lip-sync + blinks + arm
  motion, stable background.

## Setup (one time)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\echomimic_setup.ps1
```
Clones `antgroup/echomimic_v3`, builds the `echomimic_v3` conda env, installs a
**pinned** dependency set (see gotchas), downloads ~24 GB of weights into
`%USERPROFILE%\EchoMimicV3\models\`, and converts the wav2vec weights to
safetensors. Needs conda + git + ffmpeg on PATH.

| Local path | Source | Provides |
|---|---|---|
| `models\Wan2.1-Fun-V1.1-1.3B-InP` | HF `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP` | VAE, umt5-xxl text encoder, CLIP, tokenizer, base DiT |
| `models\EchoMimicV3\echomimicv3-flash-pro\diffusion_pytorch_model.safetensors` | HF `BadToBest/EchoMimicV3` | Flash-Pro audio transformer weights |
| `models\chinese-wav2vec2-base` | HF `TencentGameMate/chinese-wav2vec2-base` | audio encoder |

## Generate a video
```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_video.ps1 `
  -Image face.jpg -Audio speech.wav -Output out.mp4
```
Defaults to the **low-RAM driver** (`scripts\infer_lowram.py`). Options:
`-Prompt`, `-Steps` (8 head / 15-25 body), `-VideoLength` (seconds x 25),
`-PartialLength` / `-Overlap` (chunking, see below), `-SampleSize`,
`-GuidanceScale`, `-AudioGuidanceScale`, `-WeightDtype float16` (pre-Ampere),
`-MemMode sequential_cpu_offload` (if 12 GB VRAM is tight),
`-Driver flash` (upstream path, needs ~24 GB RAM).

## Long clips: chunked generation
The model's window is ~one chunk (81 frames / ~3.2 s); past ~138 frames a
single-shot pass degrades or OOMs. For `-VideoLength` beyond `-PartialLength`,
`infer_lowram.py` generates overlapping `PartialLength`-frame chunks, each
conditioned on the tail `Overlap` frames of the previous one (via
`get_image_to_video_latent3`, which takes a list of continuation frames), then
linearly cross-fades the overlap. Ported from upstream `app_mm.py`. Time scales
~linearly per chunk (15 s = 5 chunks ~= 8 min on the 4070). Motion stays
re-anchored so it doesn't drift, but per-chunk expression variety is bounded by
the 81-frame window - raise `-Steps` for more motion at a time cost.

## Controlling head motion
There's no explicit pose control. Big horizontal head turns (yaw) force the
model to invent a profile it can't get from a frontal photo, so face detail
smears. To keep the head frontal (nod / vertical tilt only), combine:
- a prompt that spells it out ("keeps his head facing forward toward the camera
  the entire time; only gentle vertical nodding; does not turn his head left or
  right");
- `-NegativePrompt "turning head to the side, head rotation, profile view,
  three-quarter view, side of face, face distortion, loss of facial detail"`
  (prepended to the driver's hand/finger defaults);
- `-GuidanceScale 7.5` (up from 6.0) so it adheres harder. Higher still (8.5+)
  clamps motion more but trends toward stiff.
Verified: on the Prabhupada 15 s clip this removed the yaw swings v1 had at
~5-8 s while keeping lip-sync and vertical nods.

## Why a custom low-RAM driver
Upstream `infer_flash.py` loads text encoder (11.4 GB) + CLIP (4.7 GB) +
transformer (3 GB) + flash weights (3.7 GB) + VAE all into system RAM *before*
offloading to GPU — ~23 GB resident, which crashes a 16 GB box mid-load.

`scripts\infer_lowram.py` loads each big module, uses it, and frees it before
the next: tokenizer+text-encoder -> encode prompt -> free; CLIP -> encode image
-> free; then transformer + VAE + a stubbed pipeline (`encode_prompt` /
`clip_image_encoder` monkeypatched to return the precomputed tensors). Peak RAM
~12 GB. It also fixes the upstream bug below for free.

## Windows / dependency gotchas (all handled by the scripts)
- **`--GPU_memory_mode` is dead code in `infer_flash.py`** — parsed, never
  applied (always `pipeline.to(device)`). `scripts\patch_infer_flash.py` wires
  it to the pipeline's `enable_*_cpu_offload` (only used by `-Driver flash`).
- **`requirements.txt` bare lower bounds resolve wrong.** Pinned set that works:
  `torch==2.5.1+cu121`, `transformers==4.51.3`, `diffusers==0.32.2`,
  `huggingface_hub==0.30.2`, `tokenizers==0.21.4`.
  - transformers 5.x removed per-layer hidden states from `Wav2Vec2Encoder`;
    EchoMimicV3's audio features are `hidden_states[1:]` — hard break.
  - diffusers 0.40 wants `huggingface_hub` 1.x, incompatible with transformers 4.x.
  - `pip install -r requirements.txt` pulls a CPU `torch`, clobbering CUDA —
    reinstalled `--force-reinstall --no-deps` from the cu121 index afterwards.
- **`pyloudnorm`** is imported but absent from `requirements.txt` — added.
- **wav2vec `.bin` load blocked** by transformers (CVE-2025-32434) unless
  torch>=2.6 or safetensors. Setup converts `pytorch_model.bin` ->
  `model.safetensors` once.
- **Anaconda channel ToS** now gates `conda create` on defaults — env is built
  from `conda-forge` (`-c conda-forge --override-channels`).
- `xformers` / flash-attn: not needed for single-GPU (the `xfuser` imports in
  `src/dist/__init__.py` are wrapped in try/except).
- `tensorflow==2.15.0` / `retina-face` from requirements: only used by
  `app_mm.py`, not by either inference driver. CPU wheels install fine.

## Not done / open
- Only tested at 768 / 81 frames / 8 steps on demo assets. Longer clips, higher
  step counts, and non-demo inputs not yet exercised.
- `model_cpu_offload` (~30 s/step) vs `sequential_cpu_offload` (slower, less
  VRAM) not benchmarked against each other on this box.
- The `mmgp`-quantized `app_mm.py` path was not pursued (low-RAM driver made it
  unnecessary).
- Pagefile: setup box has a 32 GB system-managed pagefile; a larger fixed one
  would add headroom for `-Driver flash` but needs an elevated shell + reboot.
