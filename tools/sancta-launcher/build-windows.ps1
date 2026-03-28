#Requires -Version 5.1
<#
  Ensures PATH for Go + gcc, then: go mod tidy && go build -o sancta-launcher.exe .
  Uses -ldflags "-H windowsgui" so double-clicking does not open a black CMD window.
  Run from tools/sancta-launcher (or pass -Root).

  For a console build (stdio / easier CLI debugging): -NoWindowsGUI
#>
param(
    [string] $Root = $PSScriptRoot,
    [switch] $NoWindowsGUI
)

Set-Location $Root
$ErrorActionPreference = "Stop"

# Reuse same PATH prep as setup (Internal = do not exit the host process)
$setupOk = & "$PSScriptRoot\setup-windows-build.ps1" -Internal
# Require explicit $true (go/gcc output must use Out-Host in setup, not success pipeline)
if ($setupOk -ne $true) { exit 1 }

$env:CGO_ENABLED = "1"
go mod tidy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# -H windowsgui = Windows GUI subsystem (no extra CMD when starting the .exe from Explorer).
if ($NoWindowsGUI) {
    go build -ldflags "-s -w" -o sancta-launcher.exe .
} else {
    go build -ldflags "-s -w -H windowsgui" -o sancta-launcher.exe .
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built: $(Join-Path $Root 'sancta-launcher.exe')" -ForegroundColor Green
