param([string]$Serial)

$ErrorActionPreference = 'Stop'

$adbCommand = Get-Command adb -ErrorAction SilentlyContinue
if ($null -eq $adbCommand) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
    if (-not (Test-Path -LiteralPath $fallback)) {
        throw 'ADB was not found. Install Android SDK Platform Tools or Meta Quest Developer Hub.'
    }
    $adb = $fallback
} else {
    $adb = $adbCommand.Source
}

& $adb start-server | Out-Host
$deviceLines = @(& $adb devices -l)
$authorizedQuests = @($deviceLines | Where-Object {
    $_ -match "\sdevice\s" -and $_ -match "(?:product:eureka|model:Quest_)"
})
if ($Serial) {
    $escapedSerial = [regex]::Escape($Serial)
    $authorizedQuests = @($authorizedQuests | Where-Object { $_ -match "^$escapedSerial\s" })
}
if ($authorizedQuests.Count -eq 0) {
    $deviceLines | Out-Host
    throw 'No authorized Quest was found. Enable Developer Mode, reconnect a data-capable USB cable, and accept Allow USB debugging inside the headset.'
}
if ($authorizedQuests.Count -gt 1) {
    $authorizedQuests | Out-Host
    throw 'Multiple authorized Quests were found. Run this script with -Serial <quest-serial>.'
}

$questSerial = ($authorizedQuests[0] -split '\s+')[0]

& $adb -s $questSerial reverse tcp:8443 tcp:8443 | Out-Host
Write-Host "Quest $questSerial USB forwarding is ready. Open http://localhost:8443/ in the Quest Browser." -ForegroundColor Green
