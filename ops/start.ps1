# 감독자와 모니터를 각각 별도 창으로 띄운다. 이미 떠 있으면(pid 파일의 프로세스가 살아 있으면) 다시 띄우지 않는다.
# 작업 스케줄러가 30분마다 이 스크립트를 부르므로, 어느 창이 닫혀도 감독자는 되살아나 체크포인트에서 이어 간다.
# 사용법:
#   powershell -ExecutionPolicy Bypass -File ops/start.ps1 -Python E:\venvs\vlm_coldstart\Scripts\python.exe
param(
    [string]$Python = "python",
    [string]$Queue = "ops/runtime/queue.json",
    [double]$StallMinutes = 20,
    [int]$MaxAttempts = 2,
    [switch]$NoMonitor
)
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:PYTHONUTF8 = "1"

function Alive([string]$pidFile) {
    if (-not (Test-Path $pidFile)) { return $false }
    $id = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $id) { return $false }
    $p = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
    return ($null -ne $p -and $p.ProcessName -like "python*")
}

if (Alive "$root\ops\runtime\supervisor.pid") {
    Write-Output "감독자 이미 실행 중"
} else {
    $sup = Start-Process -FilePath $Python -WorkingDirectory $root -PassThru -WindowStyle Minimized `
        -ArgumentList "-m", "ops.supervisor", "--queue", $Queue, "--python", $Python, `
        "--stall-minutes", $StallMinutes, "--max-attempts", $MaxAttempts
    Write-Output "감독자 시작 pid $($sup.Id)"
}
if (-not $NoMonitor) {
    if (Alive "$root\ops\runtime\monitor.pid") {
        Write-Output "모니터 이미 실행 중"
    } else {
        $mon = Start-Process -FilePath $Python -WorkingDirectory $root -PassThru -ArgumentList "-m", "ops.monitor"
        Write-Output "모니터 시작 pid $($mon.Id)"
    }
}
