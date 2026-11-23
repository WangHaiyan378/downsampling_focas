# =====================================================================
# 一鍵訓練腳本（PowerShell）
# 用法：
#   .\scripts\run_train.ps1                       # 預設跑 task_b_wind10m
#   .\scripts\run_train.ps1 -Config task_a_wind10m -Epochs 100
#   .\scripts\run_train.ps1 -Config task_b_ssrd -Demo -Epochs 3
#   .\scripts\run_train.ps1 -Config task_b -Device cuda:0
# =====================================================================

param(
    [string]$Config   = "task_b_wind10m",     # configs/<Config>.yaml
    [int]$Epochs      = 0,                    # 0 = 用 yaml 內預設值
    [int]$BatchSize   = 0,
    [double]$Lr       = 0,
    [string]$Device   = "cuda",
    [switch]$Demo
)

$ErrorActionPreference = "Stop"
$Python = "D:\miniconda\envs\yiti3\python.exe"
$Repo   = "D:\downsampling_focas"
$ConfigPath = Join-Path $Repo "configs\$Config.yaml"

if (-not (Test-Path $ConfigPath)) {
    Write-Error "找不到 config: $ConfigPath"
}

$cmd = @($Python, "main.py", "train", "--config", $ConfigPath, "--device", $Device)
if ($Epochs    -gt 0) { $cmd += @("--epochs",     "$Epochs") }
if ($BatchSize -gt 0) { $cmd += @("--batch_size", "$BatchSize") }
if ($Lr        -gt 0) { $cmd += @("--lr",         "$Lr") }
if ($Demo)            { $cmd += "--demo" }

Write-Host "[run_train] $($cmd -join ' ')" -ForegroundColor Cyan
Push-Location $Repo
try {
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Training exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "[run_train] 完成。checkpoint 位於 checkpoints/$Config/best_model.pt" -ForegroundColor Green
