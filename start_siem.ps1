# Start Sancta SIEM server - keeps window open to show errors if crash occurs
Set-Location $PSScriptRoot
# Disable safe modes so live events and agent activity are shown (set to "true" if crashes occur)
$env:SIEM_METRICS_SAFE_MODE = "false"
$env:SIEM_WS_SAFE_MODE = "false"
Write-Host "Starting SIEM at http://127.0.0.1:8787 ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
Write-Host "`nServer stopped. Press any key to close." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
