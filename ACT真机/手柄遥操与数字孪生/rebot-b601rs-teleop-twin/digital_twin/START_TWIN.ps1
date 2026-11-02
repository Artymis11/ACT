param([Parameter(Mandatory=$true)][string]$PythonPath,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$viewerScript=Join-Path $PSScriptRoot 'server.py'
$viewerLogs=Join-Path $PSScriptRoot 'logs'
[void](New-Item -ItemType Directory -Path $viewerLogs -Force)
$viewerUrl=$null
foreach ($viewerPort in 8767..8777) {
    $candidateUrl="http://127.0.0.1:$viewerPort"
    try {
        $existing=Invoke-RestMethod -Uri "$candidateUrl/api/state" -TimeoutSec 1
        if ($existing.app_id -eq 'b601-feedback-twin-public-v1') { $viewerUrl=$candidateUrl;break }
        continue
    } catch {
        if (Get-NetTCPConnection -LocalPort $viewerPort -State Listen -ErrorAction SilentlyContinue) { continue }
    }
    $arguments=@('-B','-X','utf8',('"'+$viewerScript+'"'),'--port',"$viewerPort")
    $process=Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $viewerLogs "server_$viewerPort.log") -RedirectStandardError (Join-Path $viewerLogs "server_$viewerPort.error.log") -PassThru
    for ($attempt=0;$attempt -lt 25;$attempt++) {
        Start-Sleep -Milliseconds 200
        try {
            $state=Invoke-RestMethod -Uri "$candidateUrl/api/state" -TimeoutSec 1
            if ($state.app_id -eq 'b601-feedback-twin-public-v1') { $viewerUrl=$candidateUrl;break }
        } catch {}
        if ($process.HasExited) { break }
    }
    if ($viewerUrl) { break }
}
if (-not $viewerUrl) { throw "显示服务启动失败，请查看 $viewerLogs" }
Write-Host "数字孪生：$viewerUrl"
if (-not $NoBrowser) { Start-Process $viewerUrl }
