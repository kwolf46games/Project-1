@echo off
title Test drafted answers
echo.
echo  Checking your key and connection...
echo.
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo  Could not find the .venv folder next to this file, so Python could not start.
  goto :end
)
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py" generate --check
:end
echo.
echo  When you have read the messages above, press a key to close this window.
pause >nul
