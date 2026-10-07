<#
.SYNOPSIS
  Deploy the Gradio Space (and optionally the model repo) straight from this
  checkout — no web uploads.

.EXAMPLE
  # One-time login (paste a token with WRITE access from
  # huggingface.co/settings/tokens):
  hf auth login

  # Every deploy after that:
  $env:HF_SPACE_REPO = "Ashish-04007/voice-ai-detector"
  .\spaces\deploy_space.ps1

  # First ever run (also pushes the 363 MB weights to the model repo):
  $env:HF_MODEL_REPO = "Ashish-04007/voice-ai-detector-model"
  .\spaces\deploy_space.ps1 -PushModel
#>
param(
  [switch]$PushModel
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$space = $env:HF_SPACE_REPO
if (-not $space) {
  throw "Set `$env:HF_SPACE_REPO first, e.g. `$env:HF_SPACE_REPO = 'Ashish-04007/voice-ai-detector'"
}

function Upload-ToHub([string]$repo, [string]$local, [string]$remote, [string]$type, [string]$msg) {
  Write-Host ">> $local -> $repo::$remote"
  hf upload $repo $local $remote --repo-type=$type --commit-message $msg
}

$stamp = "deploy $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Upload-ToHub $space "$root\spaces\gradio\app.py" "app.py" "space" "${stamp}: app"
Upload-ToHub $space "$root\spaces\gradio\requirements.txt" "requirements.txt" "space" "${stamp}: deps"
Upload-ToHub $space "$root\spaces\gradio\README.md" "README.md" "space" "${stamp}: card"

if ($PushModel) {
  $modelRepo = $env:HF_MODEL_REPO
  if (-not $modelRepo) {
    throw "Set `$env:HF_MODEL_REPO first, e.g. `$env:HF_MODEL_REPO = 'Ashish-04007/voice-ai-detector-model'"
  }
  Upload-ToHub $modelRepo "$root\model\detector.pt" "detector.pt" "model" "${stamp}: weights"
  Upload-ToHub $modelRepo "$root\model\detector_config.json" "detector_config.json" "model" "${stamp}: config"
}

$user, $name = $space -split "/", 2
Write-Host ""
Write-Host "Done. Watch the build at https://huggingface.co/spaces/$space"
