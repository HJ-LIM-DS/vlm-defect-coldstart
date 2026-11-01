# Windows 작업 스케줄러에 감독자 되살리기 작업을 등록하거나 해제한다.
# 등록하면 30분마다 ops/start.ps1 을 실행한다. 감독자가 살아 있으면 아무 일도 하지 않고, 죽어 있으면 다시 띄운다.
# 로그온한 사용자 세션에서만 실행되며 암호를 저장하지 않는다.
# 사용법:
#   powershell -ExecutionPolicy Bypass -File ops/schedule.ps1 -Register -Python E:\venvs\vlm_coldstart\Scripts\python.exe
#   powershell -ExecutionPolicy Bypass -File ops/schedule.ps1 -Unregister
#   powershell -ExecutionPolicy Bypass -File ops/schedule.ps1 -Query
param(
    [switch]$Register,
    [switch]$Unregister,
    [switch]$Query,
    [string]$Python = "python",
    [int]$Minutes = 30
)
$name = "vlm-defect-coldstart-supervisor"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($Register) {
    $cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File \`"$root\ops\start.ps1\`" -Python \`"$Python\`""
    schtasks /Create /TN $name /SC MINUTE /MO $Minutes /TR $cmd /IT /F
    schtasks /Query /TN $name /FO LIST | Select-String -Pattern "TaskName|Status|Next Run|Task To Run|Schedule Type|Repeat: Every"
}
if ($Unregister) {
    schtasks /Delete /TN $name /F
}
if ($Query) {
    schtasks /Query /TN $name /FO LIST
}
