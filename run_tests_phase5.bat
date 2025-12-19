@echo off
REM filepath: c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\run_tests_phase5.bat

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
echo =========================================
echo  Phase 5: Comparative Baselines Tests
echo  Version: 1.0
echo  Date: %date% %time%
echo =========================================
echo.

if not exist output_data mkdir output_data

set results_log=output_data\phase5_results.log
echo Phase 5 Comparative Baselines Tests - %date% %time% > %results_log%
echo. >> %results_log%

REM Test 5.1a
echo.
echo [T5.1a] Evaluating best GRU4Rec (layers=112, train_full)

python run.py output_data/T3.1_full.pt ^
  -l ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 1 5 10 20 ^
  -d cpu >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T5.1a] PASSED - Best GRU4Rec baseline
    echo [T5.1a] PASSED >> %results_log%
) else (
    echo [T5.1a] FAILED
    echo [T5.1a] FAILED >> %results_log%
)
echo.

REM Test 5.2a
echo [T5.2a] Evaluating Item Coverage on best model

python run.py output_data/T3.1_full.pt ^
  -l ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 20 ^
  -d cpu >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T5.2a] PASSED - Coverage metrics recorded
    echo [T5.2a] PASSED >> %results_log%
) else (
    echo [T5.2a] FAILED
    echo [T5.2a] FAILED >> %results_log%
)
echo.

REM Test 5.3 - Check for RetailRocket data
if not exist "input_data\retailrocket-data\retailrocket_train_full.dat" (
    echo [T5.3] WARNING: RetailRocket dataset not found
    echo [T5.3] Expected path: input_data\retailrocket-data\retailrocket_train_full.dat
    echo [T5.3] Skipping RetailRocket tests
    echo [T5.3] RetailRocket data not found >> %results_log%
    goto skip_retailrocket
)

echo [T5.3a] Training GRU4Rec on RetailRocket (cross-entropy baseline)

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=cross-entropy,layers=128,batch_size=64,n_epochs=3,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.07 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T5.3a_rr_xe.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T5.3a] PASSED - RetailRocket cross-entropy baseline
    echo [T5.3a] PASSED >> %results_log%
) else (
    echo [T5.3a] FAILED
    echo [T5.3a] FAILED >> %results_log%
)
echo.

echo [T5.3b] Training GRU4Rec on RetailRocket (BPR-max comparison)

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=bpr-max,layers=224,batch_size=80,n_epochs=3,n_sample=256,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,bpreg=1.95,elu_param=0.5 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T5.3b_rr_bprmax.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T5.3b] PASSED - RetailRocket BPR-max comparison
    echo [T5.3b] PASSED >> %results_log%
) else (
    echo [T5.3b] FAILED
    echo [T5.3b] FAILED >> %results_log%
)
echo.

:skip_retailrocket

REM Test 5.4 - Transfer Learning
if not exist "input_data\retailrocket-data\retailrocket_train_full.dat" (
    echo [T5.4] SKIPPED: RetailRocket data not available
    echo [T5.4] SKIPPED >> %results_log%
) else (
    echo [T5.4a] Testing Yoochoose-optimized config on RetailRocket
    echo [T5.4a] %date% %time% >> %results_log%
    
    python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
      -ps loss=cross-entropy,layers=112,batch_size=128,n_epochs=3,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08 ^
      -t input_data/retailrocket-data/retailrocket_test.dat ^
      -m 1 5 10 20 ^
      -d cpu ^
      -s output_data/T5.4a_rr_transfer.pt >> %results_log% 2>&1
    
    if %errorlevel% equ 0 (
        echo [T5.4a] PASSED - Transfer learning test completed
        echo [T5.4a] PASSED >> %results_log%
    ) else (
        echo [T5.4a] FAILED
        echo [T5.4a] FAILED >> %results_log%
    )
    echo.
)

echo.
echo =========================================
echo  Phase 5 Tests Completed
echo  Results logged to: %results_log%
echo =========================================
echo.

pause
exit /b