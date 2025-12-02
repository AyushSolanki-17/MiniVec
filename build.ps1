# run from repo root in Powershell
mkdir build -ErrorAction Ignore
cd build
cmake .. 
cmake --build . --config Release
Write-Host "Built. To test, set PYTHONPATH to $(Get-Location) and run python tests"
