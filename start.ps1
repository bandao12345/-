[CmdletBinding()]
param([ValidateRange(1024,65535)][int]$Port=5003,[switch]$CheckOnly,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
Set-Location -LiteralPath $projectRoot
$appUrl="http://127.0.0.1:$Port"
function Get-InklabService {
    param([string]$Url)
    try {
        $service=(Invoke-RestMethod -Uri ($Url.TrimEnd('/')+'/api/info') -TimeoutSec 2 -ErrorAction Stop).info
        if ($service.name -notlike '*INKLAB*') { return $null }
        $modeIds=@($service.supported_modes | ForEach-Object { $_.id })
        if ('chinese' -notin $modeIds -or 'alphanumeric' -notin $modeIds) { return $null }
        return $service
    } catch { return $null }
}
function Open-InklabPage([string]$Url) {
    if ($CheckOnly -or $NoBrowser) { return }
    Start-Process -FilePath $Url -ErrorAction Stop
}
function Test-Python([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try { & $Path -X utf8 -c "import sys; assert sys.version_info >= (3,11) and sys.maxsize > 2**32" 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}
function Test-Packages([string]$Path) {
    try { & $Path -X utf8 -c "import flask, PIL, numpy, cv2, rapidocr, onnxruntime, pypdf, waitress, mcp; from rapidocr import OCRVersion, ModelType; assert hasattr(OCRVersion,'PPOCRV6') and hasattr(ModelType,'MEDIUM')" 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}
function Run-Python([string]$Path,[string[]]$PythonArguments) {
    & $Path @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: exit $LASTEXITCODE. Read the error above." }
}
try {
    if (-not $CheckOnly) {
        $existingListener=Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($existingListener) {
            $runningService=Get-InklabService $appUrl
            if (-not $runningService) { throw "Port $Port is occupied by another app or an old INKLAB version. Stop its old server window, then run start.bat again." }
            if (-not $runningService.ready) { throw "INKLAB is running but its model is not ready. Run start.bat -CheckOnly to check the model." }
            Write-Host "INKLAB is already running: $appUrl/"
            Open-InklabPage "$appUrl/"
            return
        }
    }
    $localPython=Join-Path $projectRoot '.venv\Scripts\python.exe'
    $twoLevelsUp=Split-Path -Parent (Split-Path -Parent $projectRoot)
    $sharedOcrPython=Join-Path $twoLevelsUp '.venv-ocr-v6\Scripts\python.exe'
    $previousOcrPython=Join-Path $twoLevelsUp '.venv-ocr\Scripts\python.exe'
    $basePython=$null
    $chosenPython=$null
    foreach ($candidate in @($localPython,$sharedOcrPython,$previousOcrPython)) {
        if (Test-Python $candidate) {
            if (-not $basePython) { $basePython=$candidate }
            if (Test-Packages $candidate) { $chosenPython=$candidate; break }
        }
    }
    if (-not $chosenPython) {
        if ($CheckOnly) { throw 'OCR environment not ready. Run start.bat to install the separate OCR environment.' }
        if (-not $basePython) {
            $oldPython=Join-Path $twoLevelsUp '.venv\Scripts\python.exe'
            if (Test-Python $oldPython) { $basePython=$oldPython }
        }
        if (-not $basePython) {
            $launcher=Get-Command py -ErrorAction SilentlyContinue
            if ($launcher) {
                $candidate=(& $launcher.Source -3 -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
                if ($candidate -and (Test-Python $candidate)) { $basePython=$candidate }
            }
        }
        if (-not $basePython) {
            $candidate=Get-Command python -ErrorAction SilentlyContinue
            if ($candidate -and (Test-Python $candidate.Source)) { $basePython=$candidate.Source }
        }
        if (-not $basePython) { throw 'Install 64-bit Python 3.12, then run start.bat again.' }
        if (-not (Test-Python $localPython)) { Run-Python $basePython @('-m','venv',(Join-Path $projectRoot '.venv')) }
        $chosenPython=$localPython
        Run-Python $chosenPython @('-X','utf8','-m','pip','install','-r',(Join-Path $projectRoot 'requirements.txt'))
    }
    Write-Host "OCR Python: $chosenPython"
    Run-Python $chosenPython @('-X','utf8','-c',"from ocr_engine import UnifiedRecognizer; r=UnifiedRecognizer(); s=r.status(); print(s); raise SystemExit(0 if s['ready'] else 1)")
    if ($CheckOnly) { Write-Host 'OCR environment and model checks passed. No server started.'; exit 0 }
    $env:PORT=[string]$Port
    $env:PYTHONUTF8='1'
    Write-Host "Open http://127.0.0.1:$Port/ after the server starts."
    Write-Host 'Keep this window open. Press Ctrl+C to stop. The original app stays on port 5001.'
    $browserJob=$null
    if (-not $NoBrowser) {
        # The server stays in this console. A hidden job waits for real model
        # readiness, so the browser never opens before the server is listening.
        $browserJob=Start-Job -ArgumentList $appUrl,${function:Get-InklabService}.ToString() -ScriptBlock {
            param($Url,$ProbeSource)
            $probe=[scriptblock]::Create($ProbeSource)
            for ($attempt=0;$attempt -lt 60;$attempt++) {
                $service=& $probe $Url
                if ($service -and $service.ready) {
                    Start-Process -FilePath ($Url+'/') -ErrorAction Stop
                    return
                }
                Start-Sleep -Milliseconds 500
            }
            Write-Warning "Automatic page opening timed out. Open $Url/ manually after the server starts."
        }
    }
    try {
        Run-Python $chosenPython @('-X','utf8',(Join-Path $projectRoot 'serve.py'))
    } finally {
        if ($browserJob) {
            if ($browserJob.State -eq 'Running') { Stop-Job $browserJob -ErrorAction SilentlyContinue }
            Receive-Job $browserJob -ErrorAction Continue
            Remove-Job $browserJob -Force -ErrorAction SilentlyContinue
        }
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'If model files are missing, run the OCR Python with download_models.py.'
    Write-Host "If the service is running, open $appUrl/ manually."
    exit 1
}

