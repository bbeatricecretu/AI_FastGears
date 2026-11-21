# Stops every running FastGears server: python processes started with "fastgears",
# plus whatever python process is listening on port 8000.
$ids = @()
$ids += Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -match 'fastgears' } | ForEach-Object { $_.ProcessId }
$ids += Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { $_.OwningProcess }
$ids = $ids | Where-Object { $_ } | Sort-Object -Unique
$stopped = 0
foreach ($id in $ids) {
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -like 'python*') {
        Write-Host ("  stopped {0} (pid {1})" -f $p.ProcessName, $id)
        Stop-Process -Id $id -Force
        $stopped++
    }
}
if ($stopped -eq 0) { Write-Host "  nothing was running" }
