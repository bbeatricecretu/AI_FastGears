@echo off
cd /d "%~dp0"
title FastGears (restart)
echo Stopping running FastGears servers...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ids = @();" ^
  "$ids += Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Where-Object { $_.CommandLine -match 'fastgears' } | ForEach-Object { $_.ProcessId };" ^
  "$ids += Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $_.OwningProcess };" ^
  "$ids = $ids | Where-Object { $_ -and $_ -ne $PID } | Sort-Object -Unique;" ^
  "foreach ($id in $ids) { $p = Get-Process -Id $id -ErrorAction SilentlyContinue; if ($p -and $p.ProcessName -like 'python*') { Write-Host ('  stopped ' + $p.ProcessName + ' ' + $id); Stop-Process -Id $id -Force } }"
timeout /t 2 /nobreak >nul
echo Starting FastGears...
".venv\Scripts\python.exe" -m fastgears %*
pause
