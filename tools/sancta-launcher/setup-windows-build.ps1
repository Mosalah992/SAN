#Requires -Version 5.1
<#
.SYNOPSIS
  Check (and optionally install) Go + gcc for building sancta-launcher.exe on Windows.

.DESCRIPTION
  The VS Code/Cursor Go extension does NOT install the Go compiler. This script:
  - Prepends common install paths to PATH for the current session
  - Verifies `go version` and `gcc --version`
  - With -Install: runs winget for GoLang.Go and LLVM-MinGW (UCRT) when missing

.PARAMETER Install
  Run winget to install missing Go and/or LLVM-MinGW (requires winget, may prompt UAC).

.PARAMETER PersistUserPath
  If tools are found in well-known folders but not on User PATH, offer to add them (optional).

.PARAMETER Internal
  Used by build-windows.ps1: do not call exit (return boolean success to caller).

.EXAMPLE
  .\setup-windows-build.ps1
  .\setup-windows-build.ps1 -Install
#>
param(
    [switch] $Install,
    [switch] $PersistUserPath,
    [switch] $Internal
)

$ErrorActionPreference = "Stop"

function Test-Cmd($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Add-PathFront($Dir) {
    if ($Dir -and (Test-Path $Dir)) {
        $env:Path = "$Dir;$env:Path"
    }
}

# --- Session PATH: Go (often missing in new shells if installer bug) ---
Add-PathFront "C:\Program Files\Go\bin"
Add-PathFront (Join-Path $env:UserProfile "go\bin")

# --- Session PATH: LLVM-MinGW (winget extract location varies) ---
$wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
if (Test-Path $wingetRoot) {
    $foundGcc = Get-ChildItem -Path $wingetRoot -Recurse -Filter "gcc.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($foundGcc) { Add-PathFront $foundGcc.DirectoryName }
    Get-ChildItem -Path $wingetRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $bin = Join-Path $_.FullName "bin"
        if (Test-Path (Join-Path $bin "gcc.exe")) { Add-PathFront $bin }
        Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $b2 = Join-Path $_.FullName "bin"
            if (Test-Path (Join-Path $b2 "gcc.exe")) { Add-PathFront $b2 }
        }
    }
}
Add-PathFront "C:\msys64\ucrt64\bin"
Add-PathFront "C:\msys64\mingw64\bin"

if ($Install) {
    if (-not (Test-Cmd "winget")) {
        Write-Error "winget not found. Install App Installer / Windows Package Manager, or install Go and gcc manually."
    }
    if (-not (Test-Cmd "go")) {
        Write-Host "Installing Go via winget..."
        winget install --id GoLang.Go -e --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
        Add-PathFront "C:\Program Files\Go\bin"
    }
    if (-not (Test-Cmd "gcc")) {
        Write-Host "Installing LLVM MinGW (UCRT) via winget (needed for Fyne CGO)..."
        winget install --id MartinStorsjo.LLVM-MinGW.UCRT -e --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
        # Re-scan WinGet package folders
        if (Test-Path $wingetRoot) {
            Get-ChildItem -Path $wingetRoot -Recurse -Filter "gcc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object {
                Add-PathFront $_.Directory.FullName
            }
        }
    }
}

Write-Host "`n=== Checks ===" -ForegroundColor Cyan
$ok = $true
if (Test-Cmd "go") {
    # Must not write to success pipeline (would break caller: $x = & .\setup.ps1 -Internal)
    go version | Out-Host
} else {
    Write-Host "FAIL: go not on PATH. Install: https://go.dev/dl/  or: winget install GoLang.Go" -ForegroundColor Red
    $ok = $false
}
if (Test-Cmd "gcc") {
    gcc --version | Select-Object -First 1 | Out-Host
} else {
    Write-Host "FAIL: gcc not on PATH (CGO / Fyne). Install: winget install MartinStorsjo.LLVM-MinGW.UCRT  or MSYS2 gcc" -ForegroundColor Red
    $ok = $false
}

if (-not $ok) {
    Write-Host "`nAfter fixing, open a NEW PowerShell window and run:" -ForegroundColor Yellow
    Write-Host "  cd tools\sancta-launcher" 
    Write-Host "  .\build-windows.ps1"
    if ($Internal) { return $false }
    exit 1
}

if ($Internal) {
    Write-Host "`nPATH ready for go build." -ForegroundColor Green
    return $true
}
Write-Host "`nOK - run .\build-windows.ps1 from this directory to produce sancta-launcher.exe" -ForegroundColor Green
exit 0
