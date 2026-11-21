@echo off
cd /d "%~dp0"
title FastGears
".venv\Scripts\python.exe" -m fastgears %*
pause
