@echo off
setlocal
cd /d "%~dp0"
set "INKLAB_DEMO_PY="
for %%P in ("%~dp0.venv\Scripts\python.exe" "%~dp0..\..\.venv-ocr-v6\Scripts\python.exe" "%~dp0..\..\.venv-ocr\Scripts\python.exe") do (
    if exist "%%~fP" (
        "%%~fP" -X utf8 -c "import flask, waitress, rapidocr, onnxruntime, pypdf, mcp" >nul 2>&1
        if not errorlevel 1 (
            set "INKLAB_DEMO_PY=%%~fP"
            goto run_demo
        )
    )
)
echo Python environment is not ready. Run start.bat first to prepare this project.
pause
exit /b 1
:run_demo
"%INKLAB_DEMO_PY%" -X utf8 "%~dp0tools\public_demo.py" %*
set "INKLAB_DEMO_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %INKLAB_DEMO_EXIT%
