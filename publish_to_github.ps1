param(
  [string]$Repo = "malikberrada/MomentumReversalSpectralPhaseDiagram",
  [switch]$Private
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI (gh) is required: https://cli.github.com/"
}
if (-not (Test-Path .git)) {
  git init -b main
  git config user.name "Abdelmalik Berrada"
  git add .
  git commit -m "Initial reproducible MRSPD research release"
}
$visibility = if ($Private) { "--private" } else { "--public" }
& gh repo create $Repo $visibility --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw "gh repo create failed" }
Write-Host "PUBLISHED: https://github.com/$Repo"
