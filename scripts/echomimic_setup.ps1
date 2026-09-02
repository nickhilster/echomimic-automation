# EchoMimicV3 (Flash) setup script for Windows + NVIDIA GPU (12GB+ VRAM)
# Run this in an elevated-ish PowerShell (no admin needed) on the GPU machine.
# Requires: conda (Miniconda/Anaconda) and git already installed.
#   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
#   - Git: https://git-scm.com/download/win
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File echomimic_setup.ps1

$ErrorActionPreference = "Stop"

$RepoDir = "$HOME\EchoMimicV3"
$EnvName = "echomimic_v3"

Write-Host "=== 1. Checking prerequisites ===" -ForegroundColor Cyan
foreach ($cmd in @("git","conda")) {
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
} else {
    Write-Host "Repo already exists at $RepoDir, pulling latest..."
    git -C $RepoDir pull
}

Write-Host "`n=== 3. Creating conda environment ($EnvName, Python 3.10) ===" -ForegroundColor Cyan
$envExists = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $envExists) {
    conda create -y -n $EnvName python=3.10
} else {
    Write-Host "Conda env '$EnvName' already exists, skipping creation."
}

Write-Host "`n=== 4. Installing PyTorch (CUDA 12.1) ===" -ForegroundColor Cyan
conda run -n $EnvName pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`n=== 5. Installing repo requirements ===" -ForegroundColor Cyan
Push-Location $RepoDir
conda run -n $EnvName pip install -r requirements.txt
conda run -n $EnvName pip install huggingface_hub modelscope
Pop-Location

Write-Host "`n=== 6. Downloading model weights (this is several GB, will take a while) ===" -ForegroundColor Cyan
$DownloadScript = @"
from huggingface_hub import snapshot_download
import os

targets = [
    ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "models/Wan2.1-Fun-V1.1-1.3B-InP"),
    ("BadToBest/EchoMimicV3", "models/EchoMimicV3"),
]
for repo_id, subdir in targets:
    local_dir = os.path.join(r"$RepoDir", subdir)
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("Done. NOTE: verify against the repo README that these are the exact expected checkpoint repos/paths -- HF repo layout can change.")
"@
$DownloadScript | Out-File -FilePath "$env:TEMP\echomimic_download.py" -Encoding utf8
conda run -n $EnvName python "$env:TEMP\echomimic_download.py"

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Repo: $RepoDir"
Write-Host "Conda env: $EnvName"
Write-Host ""
Write-Host "IMPORTANT NEXT STEP: open $RepoDir and inspect README.md / configs/ for the exact" -ForegroundColor Yellow
Write-Host "Flash inference entrypoint (expected: run_flash.sh or app_mm.py) and its config schema." -ForegroundColor Yellow
Write-Host "Repo layouts on fast-moving research repos change often -- confirm before your first real run." -ForegroundColor Yellow
