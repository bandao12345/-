[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 5001,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Test-UsablePython([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    try {
        & $Executable -c "import sys; assert sys.version_info >= (3,10); assert sys.maxsize > 2**32" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Test-Dependencies([string]$Executable) {
    try {
        & $Executable -c "import flask, numpy, PIL, torch" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Invoke-PythonChecked([string]$Executable, [string[]]$PythonArguments) {
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE). See the output above."
    }
}

try {
    $localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $twoLevelsUp = Split-Path -Parent (Split-Path -Parent $projectRoot)
    $sharedPython = Join-Path $twoLevelsUp '.venv\Scripts\python.exe'
    $chosenPython = $null
    $basePython = $null

    foreach ($candidatePython in @($localPython, $sharedPython)) {
        if (Test-UsablePython $candidatePython) {
            if (-not $basePython) { $basePython = $candidatePython }
            if (Test-Dependencies $candidatePython) {
                $chosenPython = $candidatePython
                break
            }
        }
    }

    if (-not $basePython) {
        $launcherCommand = Get-Command py -ErrorAction SilentlyContinue
        if ($launcherCommand) {
            try {
                $launcherResult = & $launcherCommand.Source -3 -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $launcherResult) {
                    $launcherPython = [string]($launcherResult | Select-Object -Last 1)
                    if (Test-UsablePython $launcherPython) { $basePython = $launcherPython }
                }
            } catch { }
        }
        if (-not $basePython) {
            foreach ($commandName in @('python', 'python3')) {
                $pythonCommand = Get-Command $commandName -ErrorAction SilentlyContinue
                if ($pythonCommand -and (Test-UsablePython $pythonCommand.Source)) {
                    $basePython = $pythonCommand.Source
                    break
                }
            }
        }
    }

    if (-not $chosenPython) {
        if (-not $basePython) {
            throw 'Install 64-bit Python 3.10 or later, then run start.bat again. Python 3.12 is the verified version.'
        }
        if ($CheckOnly) {
            throw 'Python was found, but a complete project/shared environment was not found. Run start.bat to create the project environment and install dependencies.'
        }
        if (-not (Test-UsablePython $localPython)) {
            Write-Host 'Creating project environment (.venv)...'
            Invoke-PythonChecked $basePython @('-m', 'venv', (Join-Path $projectRoot '.venv'))
        }
        $chosenPython = $localPython
        Write-Host 'Installing dependencies. The PyTorch CPU wheel avoids downloading CUDA packages.'
        Invoke-PythonChecked $chosenPython @('-m', 'pip', 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu')
        Invoke-PythonChecked $chosenPython @('-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'requirements.txt'))
        if (-not (Test-Dependencies $chosenPython)) {
            throw 'Dependency import check failed. Review the installation output above.'
        }
    }

    Write-Host "Python: $chosenPython"
    Invoke-PythonChecked $chosenPython @('-c', "import sys, torch; print('Python', sys.version.split()[0], '| PyTorch', torch.__version__, '| CPU inference')")
    $modelFile = Join-Path $projectRoot 'models\mnist.pt'
    if (-not (Test-Path -LiteralPath $modelFile -PathType Leaf)) {
        throw 'Model missing: models\mnist.pt. Restore the supplied model before starting.'
    }
    if ($CheckOnly) {
        Write-Host 'Environment and model checks passed. No server was started.'
        exit 0
    }

    $env:PORT = [string]$Port
    $env:PYTHONUTF8 = '1'
    Write-Host "Open http://127.0.0.1:$Port/ after the server starts."
    Write-Host 'Keep this window open. Press Ctrl+C to stop the application.'
    Invoke-PythonChecked $chosenPython @((Join-Path $projectRoot 'app.py'))
} catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
