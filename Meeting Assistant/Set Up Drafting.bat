@echo off
title Set up drafted answers
echo.
echo  This saves your Groq API key so the overlay can draft answers.
echo  Get a key at console.groq.com ^> API Keys. Use a NEW key, not one you have shared.
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py" setkey groq
echo.
pause
