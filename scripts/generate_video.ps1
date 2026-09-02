<#
.SYNOPSIS
  One-command EchoMimicV3 (Flash) inference:
  reference image + audio clip  ->  lip-synced, body-animated video.

.DESCRIPTION
  Two drivers:

    -Driver lowram  (default) -> scripts\infer_lowram.py
        Loads text-encoder / CLIP / transformer one at a time, freeing each
        before the next, so it fits a ~16 GB-RAM box. Verified on RTX 4070
        (12 GB) + 16 GB RAM.

    -Driver flash            -> upstream infer_flash.py (auto-patched for
        --GPU_memory_mode). Loads the whole stack into RAM first (~23 GB) -
        only for boxes with plenty of system RAM.

  Run scripts\echomimic_setup.ps1 once first (clones the repo, builds the
  echomimic_v3 conda env, downloads ~24 GB of weights).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\generate_video.ps1 `
    -Image face.jpg -Audio speech.wav -Output out.mp4

.EXAMPLE
  # talking body: more steps, a matching prompt
  .\scripts\generate_video.ps1 -Image dancer.png -Audio song.wav -Output dance.mp4 `
    -Steps 20 -Prompt "A person is singing and dancing."
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Image,
    [Parameter(Mandatory)] [string] $Audio,
    [string] $Output,
    [string] $Prompt = "A person is speaking.",
    [int]    $Steps = 8,                   # Flash-distilled default; 15-25 for talking body
    [int]    $Seed = 43,
    [int]    $VideoLength = 81,            # frames; lower (65, 49, ...) to cut VRAM/time
    [int]    $SampleSize = 768,            # square edge in px before aspect fit
    [string] $GuidanceScale = "6.0",      # text CFG  (upstream optimal 3-6)
    [string] $AudioGuidanceScale = "3.0", # audio CFG (upstream run_flash.sh value)
    [ValidateSet("bfloat16", "float16")] [string] $WeightDtype = "bfloat16",  # bf16 needs Ampere+ (RTX 30xx/40xx)
    [ValidateSet("lowram", "flash")] [string] $Driver = "lowram",
    [ValidateSet("model_cpu_offload", "sequential_cpu_offload", "cuda")] [string] $MemMode = "model_cpu_offload",
    [string] $RepoDir = "$HOME\EchoMimicV3",
    [string] $EnvName = "echomimic_v3"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "conda not found on PATH. Open an Anaconda/Miniconda PowerShell, or run scripts\echomimic_setup.ps1 first." -ForegroundColor Red
    exit 1
}

function Resolve-InputPath([string] $p, [string] $label) {
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "$label not found: $p" -ForegroundColor Red
        exit 1
    }
    return (Resolve-Path -LiteralPath $p).Path
}

$imageAbs = Resolve-InputPath $Image "Image"
$audioAbs = Resolve-InputPath $Audio "Audio"

# --- resolve output path ---
if (-not $Output) {
    $Output = Join-Path (Get-Location).Path ("{0}_output.mp4" -f [IO.Path]::GetFileNameWithoutExtension($imageAbs))
}
$outDir = Split-Path -Parent $Output
if (-not $outDir) { $outDir = (Get-Location).Path }
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$outDirAbs = (Resolve-Path -LiteralPath $outDir).Path
$outputAbs = Join-Path $outDirAbs (Split-Path -Leaf $Output)

# --- locate weights laid down by echomimic_setup.ps1 ---
if (-not (Test-Path -LiteralPath $RepoDir)) {
    Write-Host "EchoMimicV3 repo not found at $RepoDir." -ForegroundColor Red
    Write-Host "Run scripts\echomimic_setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
$modelsDir  = Join-Path $RepoDir "models"
$wanDir     = Join-Path $modelsDir "Wan2.1-Fun-V1.1-1.3B-InP"
$flashCkpt  = Join-Path $modelsDir "EchoMimicV3\echomimicv3-flash-pro\diffusion_pytorch_model.safetensors"
$wav2vecDir = Join-Path $modelsDir "chinese-wav2vec2-base"
$configYaml = Join-Path $RepoDir "config\config.yaml"

$missing = @()
foreach ($item in @(
        @{ p = $wanDir;     n = "Wan2.1-Fun base model dir" },
        @{ p = $flashCkpt;  n = "EchoMimicV3 flash-pro transformer (.safetensors)" },
        @{ p = $wav2vecDir; n = "chinese-wav2vec2-base dir" },
        @{ p = $configYaml; n = "config/config.yaml (from the cloned repo)" }
    )) {
    if (-not (Test-Path -LiteralPath $item.p)) { $missing += ("  - {0}: {1}" -f $item.n, $item.p) }
}
if ($missing.Count) {
    Write-Host "Missing model files - run scripts\echomimic_setup.ps1 first:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    exit 1
}

# Fresh work dir: both drivers skip generation if <name>_output.mp4 already
# exists in --save_path, so never point them at a reused directory.
$workDir = Join-Path $outDirAbs ("._emv3_" + [DateTime]::Now.ToString("yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

if ($Driver -eq "lowram") {
    $script = Join-Path $PSScriptRoot "infer_lowram.py"
    $inferArgs = @(
        "run", "-n", $EnvName, "--no-capture-output", "python", $script,
        "--model_name", $wanDir,
        "--transformer_path", $flashCkpt,
        "--wav2vec_model_dir", $wav2vecDir,
        "--config_path", $configYaml,
        "--image_path", $imageAbs,
        "--audio_path", $audioAbs,
        "--prompt", $Prompt,
        "--save_path", $workDir,
        "--num_inference_steps", $Steps,
        "--seed", $Seed,
        "--video_length", $VideoLength,
        "--sample_size", $SampleSize, $SampleSize,
        "--sampler_name", "Flow_Unipc",
        "--guidance_scale", $GuidanceScale,
        "--audio_guidance_scale", $AudioGuidanceScale,
        "--shift", "5.0",
        "--weight_dtype", $WeightDtype,
        "--mem_mode", $MemMode,
        "--enable_teacache",
        "--teacache_threshold", "0.1",
        "--num_skip_start_steps", "5",
        "--fps", "25"
    )
}
else {
    # upstream infer_flash.py parses --GPU_memory_mode but never applies it;
    # this idempotent patch wires it to the pipeline's cpu-offload helpers.
    $patchScript = Join-Path $PSScriptRoot "patch_infer_flash.py"
    if (Test-Path -LiteralPath $patchScript) {
        & conda run -n $EnvName --no-capture-output python $patchScript $RepoDir
        if ($LASTEXITCODE -ne 0) { Write-Host "infer_flash.py patch failed - aborting." -ForegroundColor Red; exit 1 }
    }
    $gmm = if ($MemMode -eq "cuda") { "normal" } else { $MemMode }
    $inferArgs = @(
        "run", "-n", $EnvName, "--no-capture-output", "python", "infer_flash.py",
        "--image_path", $imageAbs,
        "--audio_path", $audioAbs,
        "--prompt", $Prompt,
        "--config_path", $configYaml,
        "--model_name", $wanDir,
        "--transformer_path", $flashCkpt,
        "--wav2vec_model_dir", $wav2vecDir,
        "--save_path", $workDir,
        "--num_inference_steps", $Steps,
        "--seed", $Seed,
        "--video_length", $VideoLength,
        "--sample_size", $SampleSize, $SampleSize,
        "--sampler_name", "Flow_Unipc",
        "--guidance_scale", $GuidanceScale,
        "--audio_guidance_scale", $AudioGuidanceScale,
        "--audio_scale", "1.0",
        "--neg_scale", "1.0",
        "--neg_steps", "0",
        "--shift", "5.0",
        "--weight_dtype", $WeightDtype,
        "--GPU_memory_mode", $gmm,
        "--enable_teacache",
        "--teacache_threshold", "0.1",
        "--num_skip_start_steps", "5",
        "--riflex_k", "6",
        "--ulysses_degree", "1",
        "--ring_degree", "1",
        "--fps", "25"
    )
}

Write-Host "=== EchoMimicV3 Flash inference ($Driver / $MemMode) ===" -ForegroundColor Cyan
Write-Host "  image : $imageAbs"
Write-Host "  audio : $audioAbs"
Write-Host "  output: $outputAbs"
Write-Host ("  steps : {0}   seed: {1}   size: {2}x{2}   video_length: {3}" -f $Steps, $Seed, $SampleSize, $VideoLength)
Write-Host ""

$env:ECHOMIMIC_REPO = $RepoDir
Push-Location $RepoDir
try {
    & conda @inferArgs
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($code -ne 0) {
    Write-Host "inference exited with code $code" -ForegroundColor Red
    Remove-Item -Recurse -Force -LiteralPath $workDir -ErrorAction SilentlyContinue
    exit $code
}

$produced = Get-ChildItem -LiteralPath $workDir -Filter *_output.mp4 -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $produced) {
    Write-Host "Inference finished but no *_output.mp4 appeared in $workDir" -ForegroundColor Red
    exit 1
}
Move-Item -LiteralPath $produced.FullName -Destination $outputAbs -Force
Remove-Item -Recurse -Force -LiteralPath $workDir -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Done -> $outputAbs" -ForegroundColor Green
