param(
    [ValidateSet('all','teleop','twin','setup','check')][string]$Mode='all',
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$OutputEncoding=[Console]::OutputEncoding
$repoRoot=$PSScriptRoot

function Find-RebotPython {
    $localPython=Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { return $localPython }
    if ($env:REBOT_PYTHON -and (Test-Path -LiteralPath $env:REBOT_PYTHON)) { return $env:REBOT_PYTHON }
    $pyLauncher=Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $resolved=& $pyLauncher.Source -3.12 -c 'import sys;print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) { return ($resolved | Select-Object -Last 1) }
    }
    foreach ($name in @('python.exe','python3.exe')) {
        $candidate=Get-Command $name -ErrorAction SilentlyContinue
        if ($candidate) {
            $resolved=& $candidate.Source -c 'import sys;print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) { return ($resolved | Select-Object -Last 1) }
        }
    }
    throw '未找到 Python。请安装 Python 3.12 x64，或把 REBOT_PYTHON 设置为 python.exe 完整路径。'
}

$exitCode=1
$launcherMutex=$null
$ownsMutex=$false
Push-Location -LiteralPath $repoRoot
try {
    $python=Find-RebotPython
    & $python -c 'import sys,struct;assert sys.version_info >= (3,12), "Python 3.12+ required";assert struct.calcsize("P")==8, "x64 Python required"'
    if ($LASTEXITCODE -ne 0) { throw '需要 Python 3.12 或更高版本的64位解释器。' }
    if ($CheckOnly) { Write-Host "启动路径检查通过：$python；没有打开CAN。";exit 0 }
    if ($Mode -eq 'setup') {
        & $python -B -X utf8 (Join-Path $repoRoot 'tools\bootstrap.py')
        if ($LASTEXITCODE -ne 0) { throw '资源准备失败，请查看上述具体信息。' }
        & $python -B -X utf8 (Join-Path $repoRoot 'tools\check.py')
        exit $LASTEXITCODE
    }
    if ($Mode -eq 'check') {
        & $python -B -X utf8 (Join-Path $repoRoot 'tools\check.py')
        exit $LASTEXITCODE
    }
    $verifyArgs=@('-B','-X','utf8',(Join-Path $repoRoot 'tools\bootstrap.py'),'--verify')
    if ($Mode -eq 'teleop') { $verifyArgs+='--teleop-only' }
    if ($Mode -eq 'twin') { $verifyArgs+='--twin-only' }
    & $python @verifyArgs
    if ($LASTEXITCODE -ne 0) { throw '首次使用请先运行“首次准备与检查.cmd”。' }
    if ($Mode -eq 'all' -or $Mode -eq 'twin') {
        & (Join-Path $repoRoot 'digital_twin\START_TWIN.ps1') -PythonPath $python
        if ($Mode -eq 'twin') { exit 0 }
    }
    $launcherMutex=New-Object System.Threading.Mutex($false,'Local\RebotB601TonyTeleopLaunch')
    try { $ownsMutex=$launcherMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $ownsMutex=$true }
    if (-not $ownsMutex) { throw '已有遥操窗口，请使用原窗口，避免重复启动控制器。' }
    $existing=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(tony_teleop|live_runner)\.py'
    })
    if ($existing.Count) { throw "已有遥操程序在运行（PID：$($existing.ProcessId -join ', ')），请使用原控制窗口。此脚本不会终止或替换它。" }
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'teleop\vendor\pcan\PCANBasic.dll'))) {
        throw '真实控制缺少PCANBasic.dll。请按docs/01-设备与环境.md注册官方x64 DLL；离线测试及数字孪生可单独使用。'
    }
    $reportDir=Join-Path $repoRoot 'teleop\reports'
    [void](New-Item -ItemType Directory -Path $reportDir -Force)
    $report=Join-Path $reportDir ('teleop_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'.json')
    Write-Host '等待 X：松开 LB、摇杆/方向键/扳机回中，再新按 X。初始25%速度。'
    Write-Host 'A长按3秒回本次起点后失能；B慢回零失能；Y只回本次起点。'
    & $python -B -X utf8 (Join-Path $repoRoot 'teleop\tony_teleop.py') --control --allow-enable --initial-speed-level 0 --out $report
    $exitCode=$LASTEXITCODE
}
catch { Write-Host $_.Exception.Message -ForegroundColor Red;$exitCode=1 }
finally {
    if ($ownsMutex) { $launcherMutex.ReleaseMutex() }
    if ($null -ne $launcherMutex) { $launcherMutex.Dispose() }
    Pop-Location
}
exit $exitCode
