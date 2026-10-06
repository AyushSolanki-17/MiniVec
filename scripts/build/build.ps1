$RootDir = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$BuildDir = Join-Path $RootDir 'build'
cmake -S (Join-Path $RootDir 'cpp') -B $BuildDir -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Build complete: $BuildDir"
