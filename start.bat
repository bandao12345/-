@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "INKLAB_EXIT=%ERRORLEVEL%"
if not "%INKLAB_EXIT%"=="0" pause
exit /b %INKLAB_EXIT%

