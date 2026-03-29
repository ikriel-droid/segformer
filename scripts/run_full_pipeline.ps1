param(
    [string]$Category = "bottle",
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",
    [switch]$SkipDeps,
    [switch]$SkipFetch,
    [switch]$SkipRefiner
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$PythonExe = "python"
$DepsDir = Join-Path $ProjectRoot ".deps"
$OutputRoot = Join-Path $ProjectRoot "outputs\pipeline\$Category\$Mode"
$CheckpointDir = Join-Path $OutputRoot "checkpoints"
$LogDir = Join-Path $OutputRoot "logs"
$SummaryPath = Join-Path $OutputRoot "PIPELINE_SUMMARY.md"
$ConfigTemplate = if ($Mode -eq "smoke") {
    Join-Path $ProjectRoot "examples\mvtec_bottle_smoke.toml"
}
else {
    Join-Path $ProjectRoot "examples\mvtec_bottle.toml"
}
$GeneratedConfig = Join-Path $OutputRoot "config.toml"
$DatasetRoot = "data/cityscapes_mvtec_v2/$Category"
$FetchOutputRoot = "data/cityscapes_mvtec_v2"
$PatchCheckpoint = Join-Path $CheckpointDir "patch_last.pt"
$RefinerCheckpoint = Join-Path $CheckpointDir "refiner_last.pt"
$CalibrationJson = Join-Path $CheckpointDir "calibration.json"
$PredictionsJson = Join-Path $OutputRoot "predictions.json"
$EvaluationJson = Join-Path $OutputRoot "evaluation.json"
$MaskDir = Join-Path $OutputRoot "masks"
$TestImageDir = Join-Path $ProjectRoot "$DatasetRoot\leftImg8bit\test\$Category"
$EffectiveSampleThreshold = $null

New-Item -ItemType Directory -Force $OutputRoot | Out-Null
New-Item -ItemType Directory -Force $CheckpointDir | Out-Null
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command,
        [string]$LogFile
    )

    Write-Host "==> $Name" -ForegroundColor Cyan
    $exe = $Command[0]
    $args = @()
    if ($Command.Length -gt 1) {
        $args = $Command[1..($Command.Length - 1)]
    }
    & $exe @args 2>&1 | Tee-Object -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function New-GeneratedConfig {
    param(
        [string]$TemplatePath,
        [string]$DestinationPath
    )

    $configText = Get-Content $TemplatePath -Raw
    $datasetRootUnix = $DatasetRoot.Replace("\", "/")
    $checkpointDirUnix = ("outputs/pipeline/{0}/{1}/checkpoints" -f $Category, $Mode).Replace("\", "/")
    $configText = $configText -replace 'root = ".*?"', ('root = "{0}"' -f $datasetRootUnix)
    $configText = $configText -replace 'checkpoint_dir = ".*?"', ('checkpoint_dir = "{0}"' -f $checkpointDirUnix)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($DestinationPath, $configText, $utf8NoBom)
}

Push-Location $ProjectRoot
try {
    if (-not $SkipDeps) {
        $depsProbe = Join-Path $DepsDir "torch"
        if (-not (Test-Path $depsProbe)) {
            Invoke-Step -Name "Install dependencies" `
                -Command @(
                    $PythonExe, "-m", "pip", "install",
                    "--target", $DepsDir,
                    "numpy", "opencv-python", "torch", "torchvision"
                ) `
                -LogFile (Join-Path $LogDir "01_install_deps.log")
        }
        else {
            "Dependencies already present in .deps" | Tee-Object -FilePath (Join-Path $LogDir "01_install_deps.log") | Out-Null
        }
    }

    if (-not $SkipFetch) {
        Invoke-Step -Name "Fetch and convert public dataset" `
            -Command @(
                $PythonExe,
                (Join-Path $ProjectRoot "scripts\fetch_mvtec_ad.py"),
                "--category", $Category,
                "--output-root", $FetchOutputRoot
            ) `
            -LogFile (Join-Path $LogDir "02_fetch_dataset.log")
    }

    New-GeneratedConfig -TemplatePath $ConfigTemplate -DestinationPath $GeneratedConfig
    "Generated config: $GeneratedConfig" | Tee-Object -FilePath (Join-Path $LogDir "03_generate_config.log") | Out-Null

    Invoke-Step -Name "Train patch classifier" `
        -Command @(
            $PythonExe,
            (Join-Path $ProjectRoot "run_cli.py"),
            "train-patch",
            "--config", $GeneratedConfig
        ) `
        -LogFile (Join-Path $LogDir "04_train_patch.log")

    Invoke-Step -Name "Calibrate sample threshold" `
        -Command @(
            $PythonExe,
            (Join-Path $ProjectRoot "run_cli.py"),
            "calibrate",
            "--config", $GeneratedConfig,
            "--checkpoint", $PatchCheckpoint,
            "--output-json", $CalibrationJson
        ) `
        -LogFile (Join-Path $LogDir "05_calibrate.log")

    $calibration = Get-Content $CalibrationJson -Raw | ConvertFrom-Json
    $EffectiveSampleThreshold = [double]$calibration.recommended_sample_threshold

    if (-not $SkipRefiner) {
        Invoke-Step -Name "Train suspicious-tile refiner" `
            -Command @(
                $PythonExe,
                (Join-Path $ProjectRoot "run_cli.py"),
                "train-refiner",
                "--config", $GeneratedConfig,
                "--checkpoint", $PatchCheckpoint
            ) `
            -LogFile (Join-Path $LogDir "06_train_refiner.log")
    }

    $predictCommand = @(
        $PythonExe,
        (Join-Path $ProjectRoot "run_cli.py"),
        "predict-dir",
        "--config", $GeneratedConfig,
        "--checkpoint", $PatchCheckpoint,
        "--image-dir", $TestImageDir,
        "--output-json", $PredictionsJson
    )
    if ((-not $SkipRefiner) -and (Test-Path $RefinerCheckpoint)) {
        $predictCommand += @("--refiner-checkpoint", $RefinerCheckpoint, "--mask-dir", $MaskDir)
    }
    if ($null -ne $EffectiveSampleThreshold) {
        $predictCommand += @("--sample-threshold", $EffectiveSampleThreshold.ToString("0.######", [System.Globalization.CultureInfo]::InvariantCulture))
    }

    Invoke-Step -Name "Run directory inference on test split" `
        -Command $predictCommand `
        -LogFile (Join-Path $LogDir "07_predict_dir.log")

    Invoke-Step -Name "Evaluate sample metrics on test split" `
        -Command @(
            $PythonExe,
            (Join-Path $ProjectRoot "run_cli.py"),
            "evaluate-split",
            "--config", $GeneratedConfig,
            "--checkpoint", $PatchCheckpoint,
            "--split", "test",
            "--calibration-json", $CalibrationJson,
            "--output-json", $EvaluationJson
        ) `
        -LogFile (Join-Path $LogDir "08_evaluate_split.log")

    $summary = @(
        "# Pipeline Summary",
        "",
        "- Category: $Category",
        "- Mode: $Mode",
        "- Config: $GeneratedConfig",
        "- Dataset root: $DatasetRoot",
        "- Patch checkpoint: $PatchCheckpoint",
        "- Calibration json: $CalibrationJson",
        "- Effective sample threshold: $(if ($null -ne $EffectiveSampleThreshold) { $EffectiveSampleThreshold } else { "config default" })",
        "- Refiner checkpoint: $(if (Test-Path $RefinerCheckpoint) { $RefinerCheckpoint } else { "not generated" })",
        "- Predictions json: $PredictionsJson",
        "- Evaluation json: $EvaluationJson",
        "- Mask dir: $(if (Test-Path $MaskDir) { $MaskDir } else { "not generated" })",
        "",
        "## Automated steps completed",
        "",
        "1. Verified or installed Python dependencies into .deps",
        "2. Downloaded and converted public MVTec AD to Cityscapes semantic format",
        "3. Generated a run-specific config",
        "4. Trained the patch classifier",
        "5. Calibrated the sample threshold on the validation split",
        "6. Trained the refiner",
        "7. Ran prediction on the test directory and exported artifacts",
        "8. Evaluated sample-level metrics on the test split",
        "",
        "## Remaining work to reach fuller completion",
        "",
        "1. Replace smoke mode with -Mode full and train longer on the full split.",
        "2. Review calibration.json on a harder OK validation set and adjust thresholds for your FP budget.",
        "3. Add alignment templates or keypoints if your real line has pose drift.",
        "4. Swap public MVTec with your in-domain Cityscapes-format data once available.",
        "5. Validate zone-specific thresholds and hard-OK replay with real production false positives."
    ) -join "`r`n"
    Set-Content -Path $SummaryPath -Value $summary -Encoding UTF8
    Write-Host "Pipeline finished. Summary: $SummaryPath" -ForegroundColor Green
}
finally {
    Pop-Location
}
