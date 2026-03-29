@echo off
setlocal

set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_full_pipeline.ps1" %*
exit /b %ERRORLEVEL%
