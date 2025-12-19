@echo off
REM filepath: c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\run_tests_phase4.bat

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
echo =========================================
echo  Phase 4: Production Readiness Tests
echo  Version: 1.0
echo  Date: %date% %time%
echo =========================================
echo.

if not exist output_data mkdir output_data

set results_log=output_data\phase4_results.log
echo Phase 4 Production Readiness Tests - %date% %time% > %results_log%
echo. >> %results_log%

REM Test 4.1a
echo.
echo [T4.1a] Loading best model (layers=112) and evaluating latency...

python run.py output_data/T2.2_layers112.pt ^
  -l ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 1 5 10 20 ^
  -d cpu >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T4.1a] PASSED
    echo [T4.1a] PASSED >> %results_log%
) else (
    echo [T4.1a] FAILED
    echo [T4.1a] FAILED >> %results_log%
)
echo.

REM Test 4.1b
echo [T4.1b] Speed-optimized model (layers=96) latency test...

python run.py output_data/T2.2_layers96.pt ^
  -l ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 1 5 10 20 ^
  -d cpu >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T4.1b] PASSED
    echo [T4.1b] PASSED >> %results_log%
) else (
    echo [T4.1b] FAILED
    echo [T4.1b] FAILED >> %results_log%
)
echo.

REM Test 4.1c
echo [T4.1c] Production model (train_full) latency test...

python run.py output_data/T3.1_full.pt ^
  -l ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 1 5 10 20 ^
  -d cpu >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T4.1c] PASSED
    echo [T4.1c] PASSED >> %results_log%
) else (
    echo [T4.1c] FAILED
    echo [T4.1c] FAILED >> %results_log%
)
echo.

echo.
echo =========================================
echo  Phase 4 Tests Completed
echo  Results logged to: %results_log%
echo =========================================
echo.

pause
exit /b