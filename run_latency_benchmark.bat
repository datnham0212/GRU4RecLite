@echo off
REM Latency Benchmark Script for GRU4Rec

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo.
echo ===================================================
echo  Inference Latency Benchmark
echo  Date: %date% %time%
echo ===================================================
echo.

set output_file=output_data\latency_benchmark.txt
set test_data=input_data/yoochoose-data/yoochoose_test.dat

REM Clear previous results
echo Latency Benchmark Results - %date% %time% > %output_file%

REM Test 1: layers=96 (speed optimized)
echo [Benchmark] Testing layers=96 (speed optimized)... 
python benchmark_latency.py output_data/T2.2_layers96.pt ^
  -t %test_data% ^
  -n 1000 ^
  -d cpu ^
  -o %output_file%
echo.

REM Test 2: layers=112 (balanced)
echo [Benchmark] Testing layers=112 (balanced)...
python benchmark_latency.py output_data/T2.2_layers112.pt ^
  -t %test_data% ^
  -n 1000 ^
  -d cpu ^
  -o %output_file%
echo.

REM Test 3: train_full (production)
echo [Benchmark] Testing train_full (production)...
python benchmark_latency.py output_data/T3.1_full.pt ^
  -t %test_data% ^
  -n 1000 ^
  -d cpu ^
  -o %output_file%
echo.

echo ===================================================
echo  Latency Benchmark Complete
echo  Results: %output_file%
echo ===================================================
echo.

pause
