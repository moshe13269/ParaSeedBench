param([string]$Config = "configs/main_v2.lock.yaml")
$ErrorActionPreference = "Stop"
python -u -m paraseedbench.run_v2 --config $Config
if ($LASTEXITCODE -ne 0) { throw "Benchmark stopped with exit code $LASTEXITCODE. Fix the error before resuming." }
