[CmdletBinding()]
param(
    [int]$BackendPort = 8008,
    [int]$FrontendPort = 5173,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Net.Http

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..")).Path
$uiRoot = Join-Path $repoRoot "ui"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$backendUrl = "http://127.0.0.1:$BackendPort"
$frontendUrl = "http://127.0.0.1:$FrontendPort"
$appData = $env:LOCALAPPDATA
if (-not $appData) {
    $appData = Join-Path $HOME "AppData\Local"
}
$logRoot = Join-Path $appData "MoAWorkbench\logs"

function Test-PortListening {
    param([int]$Port)
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $listener
}

function Wait-HttpReady {
    param(
        [string]$Url,
        [string]$Name
    )
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(2)
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            $response = $client.GetAsync($Url).GetAwaiter().GetResult()
            if ($response.IsSuccessStatusCode) {
                Write-Host "$Name is ready at $Url"
                $client.Dispose()
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    $client.Dispose()
    Write-Warning "$Name did not answer at $Url. Check logs in $logRoot."
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing virtual environment Python at $python. Create the venv and install requirements first."
}

if (-not (Test-Path -LiteralPath $uiRoot)) {
    throw "Missing UI directory at $uiRoot."
}

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmCommand) {
    $npmCommand = Get-Command npm -ErrorAction Stop
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

if (Test-PortListening $BackendPort) {
    Write-Host "Backend port $BackendPort is already in use; leaving it running."
} else {
    $backendOut = Join-Path $logRoot "backend.out.log"
    $backendErr = Join-Path $logRoot "backend.err.log"
    Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "workbench.api:app", "--host", "127.0.0.1", "--port", "$BackendPort") `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $backendOut `
        -RedirectStandardError $backendErr `
        -WindowStyle Hidden
    Write-Host "Started backend on $backendUrl"
}

$env:VITE_MOA_BACKEND_URL = $backendUrl
$env:VITE_MOA_FRONTEND_PORT = "$FrontendPort"

if (Test-PortListening $FrontendPort) {
    Write-Host "Frontend port $FrontendPort is already in use; leaving it running."
} else {
    $frontendOut = Join-Path $logRoot "frontend.out.log"
    $frontendErr = Join-Path $logRoot "frontend.err.log"
    Start-Process `
        -FilePath $npmCommand.Source `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") `
        -WorkingDirectory $uiRoot `
        -RedirectStandardOutput $frontendOut `
        -RedirectStandardError $frontendErr `
        -WindowStyle Hidden
    Write-Host "Started frontend on $frontendUrl"
}

Wait-HttpReady -Url "$backendUrl/api/lmstudio/status" -Name "Backend"
Wait-HttpReady -Url $frontendUrl -Name "Frontend"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}

Write-Host "MoA Workbench is available at $frontendUrl"
