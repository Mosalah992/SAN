# Build Tailwind CSS for Sanctum dashboard
# Requires Node.js and npm (run: npm install first)
Set-Location $PSScriptRoot
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  Write-Host "Node.js/npm not found. Install Node.js from https://nodejs.org/ and run:" -ForegroundColor Yellow
  Write-Host "  npm install" -ForegroundColor Cyan
  Write-Host "  npm run build:css" -ForegroundColor Cyan
  exit 1
}
npm run build:css
if ($LASTEXITCODE -eq 0) {
  Write-Host "CSS built successfully. Output: frontend/siem/styles.css" -ForegroundColor Green
} else {
  Write-Host "Build failed." -ForegroundColor Red
  exit 1
}
