# =====================================================================
# 一鍵評估腳本（PowerShell）
# 預設行為：
#   - 中國陸地 + 台灣 mask（include_taiwan = True）
#   - lead 10–45 天
#   - 同時輸出：主要指標 JSON、RMSE 空間圖、4 模式 baseline 對比表（CSV+JSON+Markdown）
#
# 用法：
#   .\scripts\run_eval.ps1                                # task_b_wind10m + 最新 best_model.pt
#   .\scripts\run_eval.ps1 -Config task_b_ssrd
#   .\scripts\run_eval.ps1 -Config task_b -MaxSamples 80 -LeadMin 1 -LeadMax 45
#   .\scripts\run_eval.ps1 -Config task_b_wind10m -RebuildMask    # 第一次切到 with_taiwan mask 建議用
#   .\scripts\run_eval.ps1 -Config task_b_wind10m -NoTaiwan        # 退回 China-only 行為
# =====================================================================

param(
    [string]$Config       = "task_b_wind10m",
    [string]$Checkpoint   = "",                         # 空字串 → 自動找 checkpoints/<Config>/best_model.pt
    [string]$OutputDir    = "eval_results",
    [int]$MaxSamples      = 50,
    [int]$LeadMin         = 10,
    [int]$LeadMax         = 45,
    [string]$LeadGroups   = "01-10,11-20,21-30,31-40",
    [string]$Device       = "auto",
    [switch]$NoTaiwan,
    [switch]$RebuildMask,
    [switch]$SkipRmseMaps,
    [switch]$Demo
)

$ErrorActionPreference = "Stop"
$Python = "D:\miniconda\envs\yiti3\python.exe"
$Repo   = "D:\downsampling_focas"
$ConfigPath = Join-Path $Repo "configs\$Config.yaml"

if (-not (Test-Path $ConfigPath)) {
    Write-Error "找不到 config: $ConfigPath"
}

if ([string]::IsNullOrEmpty($Checkpoint)) {
    $Checkpoint = Join-Path $Repo "checkpoints\$Config\best_model.pt"
}
if (-not (Test-Path $Checkpoint)) {
    Write-Warning "找不到 checkpoint: $Checkpoint  → 會以隨機權重跑（僅 demo 有意義）"
}

$cmd = @(
    $Python, "main.py", "eval",
    "--config",      $ConfigPath,
    "--checkpoint",  $Checkpoint,
    "--output_dir",  $OutputDir,
    "--max_samples", "$MaxSamples",
    "--lead_min",    "$LeadMin",
    "--lead_max",    "$LeadMax",
    "--lead_groups", $LeadGroups,
    "--device",      $Device
)
if ($NoTaiwan)     { $cmd += "--no_include_taiwan" }
if ($RebuildMask)  { $cmd += "--rebuild_china_mask" }
if ($SkipRmseMaps) { $cmd += "--skip_rmse_maps" }
if ($Demo)         { $cmd += "--demo" }

Write-Host "[run_eval] $($cmd -join ' ')" -ForegroundColor Cyan
Push-Location $Repo
try {
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "[run_eval] 完成。請查看：" -ForegroundColor Green
Write-Host "  - 主要指標 JSON      : $OutputDir\eval_report_*.json"
Write-Host "  - RMSE 空間熱力圖    : $OutputDir\rmse_spatial_maps\<task>_<var>\*.png"
Write-Host "  - 4 模式對比表（CSV）: $OutputDir\baseline_comparison\*_methods.csv"
Write-Host "  - 4 模式對比表（MD） : $OutputDir\baseline_comparison\*.md"
Write-Host "  - 逐 lead 改進率表   : $OutputDir\baseline_comparison\*_lead_wise.csv"
