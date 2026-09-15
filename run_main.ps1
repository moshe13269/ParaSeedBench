param([string]$Config = "configs/main_v2.lock.yaml")
$ErrorActionPreference = "Stop"
if (!(Test-Path $Config)) { throw "Read README.md and freeze your v2 config first: $Config" }
python -u -m paraseedbench.run_v2 --config $Config
if ($LASTEXITCODE -ne 0) { throw "Run stopped: exit $LASTEXITCODE" }
