@echo off
title Set up drafted answers
echo.
echo  This saves your Groq API key so the overlay can draft answers.
echo  Get a key at console.groq.com ^> API Keys. Use a NEW key, not one you have shared.
echo.
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo  Could not find the .venv folder next to this file, so Python could not start.
  echo  Put this file in the same folder as it.bat and try again.
  goto :end
)
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py" setkey groq
:end
echo.
echo  When you have read the messages above, press a key to close this window.
pause >nul
