param(
    [string]$ProjectRoot = ".",
    [string]$Out = ".\runs\transport-v4-japan-v10\paper-diagnostics"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
python .\v4_japan_diagnostics_export.py `
  --panel .\runs\transport-v4-japan-v10\panel.csv.gz `
  --folds .\runs\transport-v4-japan-v10\confirmatory\v4_fold_certification.csv `
  --regimes .\runs\transport-v4-japan-v10\confirmatory\v4_regimes.csv `
  --out $Out `
  --bootstrap-reps 1000 `
  --seed 20260812
if ($LASTEXITCODE -ne 0) { throw "V4 Japan diagnostics export failed" }
Write-Host "MRSPD_V4_JAPAN_PAPER_DIAGNOSTICS: PASS" -ForegroundColor Green
Write-Host "Upload these two files back to ChatGPT:" -ForegroundColor Cyan
Write-Host "  $Out\v4_article_table_exact.csv"
Write-Host "  $Out\v4_japan_tail_mean_ci.png"
Write-Host "Also upload $Out\v4_subband_diagnostics_exact.csv if you want the full fold/bin appendix." -ForegroundColor Cyan
