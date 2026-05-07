# Kaggle 一键推送 + GUI 操作指南
param(
    [string]$Lr,           # 学习率
    [string]$Epochs,       # 训练轮数
    [string]$BatchSize,    # batch 大小
    [string]$WandbMode,    # online / offline
    [string]$NumWorkers,   # 数据加载线程数
    [string]$SaveInterval, # checkpoint 保存间隔
    [switch]$Show,         # 仅查看当前参数
    [switch]$PushOnly,     # 只 push 不修改参数
    [switch]$NoPush        # 只修改参数不 push
)

$ErrorActionPreference = "Stop"
$config = "configs\kaggle_full.yaml"
$kernelId = "herely/cifar10-cnn-train2"

# ---- 1. 查看当前参数 ----
if ($Show) {
    python scripts/update_config.py --file $config --show
    exit 0
}

# ---- 2. 更新参数 ----
if (-not $PushOnly) {
    $setArgs = @()
    if ($Lr)           { $setArgs += "optim.lr=$Lr" }
    if ($Epochs)       { $setArgs += "train.epochs=$Epochs" }
    if ($BatchSize)    { $setArgs += "data.batch_size=$BatchSize" }
    if ($WandbMode)    { $setArgs += "wandb.mode=$WandbMode" }
    if ($NumWorkers)   { $setArgs += "data.num_workers=$NumWorkers" }
    if ($SaveInterval) { $setArgs += "train.save_interval=$SaveInterval" }

    if ($setArgs.Count -gt 0) {
        python scripts/update_config.py --file $config --set $setArgs
        Write-Host ""
    }
}

# ---- 3. Git push ----
if (-not $NoPush) {
    git add $config kernel.py kernel-metadata.json
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "update kaggle params"
    }
    git push
    Write-Host ""
}

# ---- 4. GUI 操作指南 ----
$green = [ConsoleColor]::Green; $yellow = [ConsoleColor]::Yellow; $cyan = [ConsoleColor]::Cyan
Write-Host "=== Kaggle GUI 操作 ===" -ForegroundColor $yellow
Write-Host ""
Write-Host "1. 打开 " -NoNewline
Write-Host "https://www.kaggle.com/code/$kernelId" -ForegroundColor $cyan
Write-Host ""
Write-Host "2. Settings 面板检查：" -ForegroundColor $green
Write-Host "   [ ] Internet    → ON"
Write-Host "   [ ] Accelerator → GPU T4 x2"
Write-Host ""
Write-Host "3. Add Data 挂载这 2 个数据集：" -ForegroundColor $green
Write-Host "   [ ] herely/cifar10-extracted    (训练数据)"
Write-Host "   [ ] herely/wandb-key            (WANDB_API_KEY, 私有)"
Write-Host ""
Write-Host "4. 点击 Run" -ForegroundColor $green
Write-Host ""
Write-Host "=== 监控 ===" -ForegroundColor $yellow
Write-Host "Wandb: https://wandb.ai/herely-peking-university/cifar10-cnn-classification"
Write-Host "Logs:  kaggle kernels logs $kernelId"
