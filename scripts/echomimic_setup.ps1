# EchoMimicV3 (Flash) setup script for Windows + NVIDIA GPU (12GB+ VRAM)
# Run this in a normal PowerShell (no admin needed) on the GPU machine.
# Requires: conda (Miniconda/Anaconda) and git already installed.
#   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
#   - Git: https://git-scm.com/download/win
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File echomimic_setup.ps1
#
# Downloads ~24 GB of weights (Flash path only). Takes a while on first run.

$ErrorActionPreference = "Stop"

$RepoDir = "$HOME\EchoMimicV3"
$EnvName = "echomimic_v3"

Write-Host "=== 1. Checking prerequisites ===" -ForegroundColor Cyan
foreach ($cmd in @("git", "conda")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Missing required tool: $cmd. Install it first, then re-run this script." -ForegroundColor Red
        exit 1
    }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found on PATH. Get it from https://www.gyan.dev/ffmpeg/builds/ (essentials build), add its bin/ to PATH, then re-run." -ForegroundColor Yellow
    Write-Host "Continuing setup anyway (you'll need ffmpeg before running inference)." -ForegroundColor Yellow
}

Write-Host "`n=== 2. Cloning EchoMimicV3 ===" -ForegroundColor Cyan
if (-not (Test-Path $RepoDir)) {
    git clone https://github.com/antgroup/echomimic_v3.git $RepoDir
}
else {
    Write-Host "Repo already exists at $RepoDir, pulling latest..."
    git -C $RepoDir pull
}

Write-Host "`n=== 3. Creating conda environment ($EnvName, Python 3.10) ===" -ForegroundColor Cyan
$envExists = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $envExists) {
    conda create -y -n $EnvName python=3.10
}
else {
    Write-Host "Conda env '$EnvName' already exists, skipping creation."
}

Write-Host "`n=== 4. Installing repo requirements ===" -ForegroundColor Cyan
Push-Location $RepoDir
conda run -n $EnvName --no-capture-output pip install -r requirements.txt
# pyloudnorm: imported by infer_flash.py / infer_lowram.py, missing from upstream requirements.txt.
conda run -n $EnvName --no-capture-output pip install modelscope pyloudnorm
Pop-Location

Write-Host "`n=== 5. Pinning a working dependency set ===" -ForegroundColor Cyan
# requirements.txt uses bare lower bounds; the resolver otherwise pulls
#   - transformers 5.x  -> removed per-layer hidden states from Wav2Vec2Encoder
#                          (EchoMimicV3's audio features are hidden_states[1:])
#   - diffusers 0.40 + huggingface_hub 1.x -> mismatched with transformers 4.x
#   - a CPU torch build -> clobbers the CUDA wheels
# These are the versions verified end-to-end on RTX 4070 (12 GB) + 16 GB RAM.
conda run -n $EnvName --no-capture-output pip install `
    "transformers==4.51.3" "diffusers==0.32.2" "huggingface_hub==0.30.2" "tokenizers==0.21.4"
conda run -n $EnvName --no-capture-output pip install --force-reinstall --no-deps `
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
conda run -n $EnvName --no-capture-output python -c "import torch;assert torch.cuda.is_available(),'CUDA not available after install';print('CUDA OK:',torch.cuda.get_device_name(0))"

Write-Host "`n=== 6. Downloading model weights (~24 GB, will take a while) ===" -ForegroundColor Cyan
# Paths consumed by infer_flash.py / scripts\generate_video.ps1:
#   models/Wan2.1-Fun-V1.1-1.3B-InP                                  -> --model_name  (VAE, T5 text encoder, CLIP, tokenizer, base DiT)
#   models/EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors  -> --transformer_path
#   models/chinese-wav2vec2-base                                     -> --wav2vec_model_dir
$DownloadScript = @'
import os
from huggingface_hub import snapshot_download

repo_dir = os.environ["EMV3_REPO_DIR"].replace("\\", "/")
models = os.path.join(repo_dir, "models")

jobs = [
    # (repo_id, local_subdir, allow_patterns)
    ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "Wan2.1-Fun-V1.1-1.3B-InP", None),
    ("BadToBest/EchoMimicV3", "EchoMimicV3", ["echomimicv3-flash-pro/*"]),
    ("TencentGameMate/chinese-wav2vec2-base", "chinese-wav2vec2-base",
     ["config.json", "preprocessor_config.json", "pytorch_model.bin"]),
]

for repo_id, subdir, allow in jobs:
    local_dir = os.path.join(models, subdir)
    print(f"\n=== {repo_id}  ->  {local_dir} ===", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow,
        local_dir_use_symlinks=False,
    )

print("\nAll downloads complete (~24 GB, Flash path).")
print("If a repo layout changed upstream, cross-check the cloned README.md.")
'@
$env:EMV3_REPO_DIR = $RepoDir
$DownloadScript | Out-File -FilePath "$env:TEMP\echomimic_download.py" -Encoding utf8
conda run -n $EnvName --no-capture-output python "$env:TEMP\echomimic_download.py"

Write-Host "`n=== 7. Converting wav2vec weights to safetensors ===" -ForegroundColor Cyan
# transformers >=4.53 refuses torch.load of a .bin unless torch>=2.6 (CVE-2025-32434);
# chinese-wav2vec2-base ships only pytorch_model.bin. Re-save it as safetensors.
$ConvScript = @'
import os
import torch
from safetensors.torch import save_file

d = os.path.join(os.environ["EMV3_REPO_DIR"], "models", "chinese-wav2vec2-base")
src = os.path.join(d, "pytorch_model.bin")
dst = os.path.join(d, "model.safetensors")
if os.path.exists(dst):
    print("model.safetensors already present")
else:
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    clean = {k: v.contiguous().clone() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(clean, dst, metadata={"format": "pt"})
    print(f"wrote {dst} ({len(clean)} tensors)")
'@
$ConvScript | Out-File -FilePath "$env:TEMP\echomimic_wav2vec_conv.py" -Encoding utf8
conda run -n $EnvName --no-capture-output python "$env:TEMP\echomimic_wav2vec_conv.py"

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Repo:      $RepoDir"
Write-Host "Conda env: $EnvName"
Write-Host ""
Write-Host "Generate a video:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\generate_video.ps1 -Image face.jpg -Audio speech.wav -Output out.mp4"
