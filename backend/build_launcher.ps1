# build_launcher.ps1 — Build sancta_launcher.exe (PowerShell)
# Usage: .\build_launcher.ps1  or  .\build_launcher.bat

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& cmd /c "build_launcher.bat"
