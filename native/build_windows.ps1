param(
    [ValidateSet("lua54", "luajit")]
    [string]$LuaAbi = "lua54",

    [string]$OutDir = "bin\windows-x64"
)

$ErrorActionPreference = "Stop"

$targetDir = Join-Path $PSScriptRoot (Join-Path $OutDir $LuaAbi)
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$out = Join-Path $targetDir "luann_opencl.dll"
$src = Join-Path $PSScriptRoot "opencl_bridge.c"
$obj = Join-Path $targetDir "opencl_bridge.obj"
$implib = Join-Path $targetDir "luann_opencl.lib"
$exp = Join-Path $targetDir "luann_opencl.exp"

$clArgs = @(
    "/nologo",
    "/LD",
    "/O2",
    "/Fo:$obj",
    $src,
    "/Fe:$out",
    "/link",
    "/NOLOGO",
    "/IMPLIB:$implib"
)

function Invoke-Cl {
    param([string[]]$Arguments)

    if (Get-Command cl -ErrorAction SilentlyContinue) {
        & cl @Arguments
        if ($LASTEXITCODE -ne 0) { throw "cl failed with exit code $LASTEXITCODE" }
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "cl is not on PATH and vswhere.exe was not found. Install Visual Studio Build Tools or run from a Developer Command Prompt."
    }

    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        throw "Visual Studio C++ tools were not found. Install the Desktop development with C++ workload."
    }

    $devCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $devCmd)) {
        throw "VsDevCmd.bat was not found at $devCmd"
    }

    $quotedArgs = $Arguments | ForEach-Object {
        '"' + ($_ -replace '"', '\"') + '"'
    }
    $cmd = '"' + $devCmd + '" -arch=x64 >nul && cl ' + ($quotedArgs -join ' ')
    cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "cl failed with exit code $LASTEXITCODE" }
}

Invoke-Cl -Arguments $clArgs
Remove-Item -LiteralPath $obj, $implib, $exp -ErrorAction SilentlyContinue

Write-Host "Built $out"
