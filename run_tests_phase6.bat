@echo off
REM filepath: c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\run_tests_phase6.bat
REM Phase 6: Advanced Optimization Tests
REM - Early stopping analysis
REM - TOP-1 loss comparison
REM - RetailRocket BPR-max tuning
REM - Latency benchmarking

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
echo ===================================================
echo  Phase 6: Advanced Optimization Tests
echo  Version: 1.0
echo  Date: %date% %time%
echo ===================================================
echo.

if not exist output_data mkdir output_data

set results_log=output_data\phase6_results.log
echo Phase 6 Advanced Optimization Tests - %date% %time% > %results_log%
echo. >> %results_log%

REM =========================================================
REM T6.4 + T6.5: EARLY STOPPING ANALYSIS
REM =========================================================
echo.
echo [PHASE 6.4-6.5] Early Stopping Analysis
echo ========================================
echo Phase 6.4-6.5: Early Stopping >> %results_log%

REM T6.5a: 3 epochs only
echo.
echo [T6.5a] Training with 3 epochs (early stopping)
echo [T6.5a] %date% %time% >> %results_log%

python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=cross-entropy,layers=112,batch_size=128,n_epochs=3,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.5a_early_stop_3ep.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.5a] PASSED
    echo [T6.5a] PASSED >> %results_log%
) else (
    echo [T6.5a] FAILED
    echo [T6.5a] FAILED >> %results_log%
)
echo. >> %results_log%

REM T6.5b: 4 epochs
echo [T6.5b] Training with 4 epochs (early stopping)
echo [T6.5b] %date% %time% >> %results_log%

python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=cross-entropy,layers=112,batch_size=128,n_epochs=4,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.5b_early_stop_4ep.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.5b] PASSED
    echo [T6.5b] PASSED >> %results_log%
) else (
    echo [T6.5b] FAILED
    echo [T6.5b] FAILED >> %results_log%
)
echo. >> %results_log%

REM =========================================================
REM T6.7 + T6.8: TOP-1 LOSS COMPARISON
REM =========================================================
echo.
echo [PHASE 6.7-6.8] TOP-1 Loss Comparison (Yoochoose)
echo ===================================================
echo Phase 6.7-6.8: TOP-1 Loss >> %results_log%

REM T6.7a: TOP-1 tuned on Yoochoose
echo.
echo [T6.7a] TOP-1 tuned on Yoochoose
echo [T6.7a] %date% %time% >> %results_log%

python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=top1,layers=128,batch_size=32,n_epochs=5,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.05,momentum=0.3 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.7a_yoo_top1_tuned.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.7a] PASSED
    echo [T6.7a] PASSED >> %results_log%
) else (
    echo [T6.7a] FAILED
    echo [T6.7a] FAILED >> %results_log%
)
echo. >> %results_log%

REM T6.7b: TOP1-max tuned on Yoochoose
echo [T6.7b] TOP1-max tuned on Yoochoose
echo [T6.7b] %date% %time% >> %results_log%

python run.py input_data/yoochoose-data/yoochoose_train_valid.dat ^
  -ps loss=top1-max,layers=128,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0 ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/T6.7b_yoo_top1max_tuned.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.7b] PASSED
    echo [T6.7b] PASSED >> %results_log%
) else (
    echo [T6.7b] FAILED
    echo [T6.7b] FAILED >> %results_log%
)
echo. >> %results_log%

REM =========================================================
REM T6.8: TOP-1 ON RETAILROCKET
REM =========================================================
echo.
echo [PHASE 6.8] TOP-1 Loss Comparison (RetailRocket)
echo =================================================
echo Phase 6.8: TOP-1 on RetailRocket >> %results_log%

if not exist "input_data/retailrocket-data/retailrocket_train_full.dat" (
    echo [T6.8] WARNING: RetailRocket data not found
    echo [T6.8] Skipping RetailRocket TOP-1 tests
    echo [T6.8] RetailRocket data not found >> %results_log%
    goto skip_rr_top1
)

REM T6.8a: TOP-1 on RetailRocket
echo.
echo [T6.8a] TOP-1 on RetailRocket
echo [T6.8a] %date% %time% >> %results_log%

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=top1,layers=100,batch_size=32,n_epochs=3,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.05,momentum=0.3 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.8a_rr_top1_tuned.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.8a] PASSED
    echo [T6.8a] PASSED >> %results_log%
) else (
    echo [T6.8a] FAILED
    echo [T6.8a] FAILED >> %results_log%
)
echo. >> %results_log%

REM T6.8b: TOP1-max on RetailRocket
echo [T6.8b] TOP1-max on RetailRocket
echo [T6.8b] %date% %time% >> %results_log%

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=top1-max,layers=100,batch_size=64,n_epochs=3,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.8b_rr_top1max_tuned.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.8b] PASSED
    echo [T6.8b] PASSED >> %results_log%
) else (
    echo [T6.8b] FAILED
    echo [T6.8b] FAILED >> %results_log%
)
echo. >> %results_log%

:skip_rr_top1

REM =========================================================
REM T6.3: RETAILROCKET BPR-MAX TUNING
REM =========================================================
echo.
echo [PHASE 6.3] RetailRocket BPR-max Tuning
echo =====================================
echo Phase 6.3: BPR-max tuning >> %results_log%

if not exist "input_data/retailrocket-data/retailrocket_train_full.dat" (
    echo [T6.3] WARNING: RetailRocket data not found
    echo [T6.3] Skipping BPR-max tuning tests
    echo [T6.3] RetailRocket data not found >> %results_log%
    goto skip_rr_bprmax
)

REM T6.3b: BPR-max tuned (5 epochs)
echo.
echo [T6.3b] BPR-max with 5 epochs and layers=192
echo [T6.3b] %date% %time% >> %results_log%

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=bpr-max,layers=192,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.05,learning_rate=0.08,momentum=0.4,bpreg=1.95,elu_param=0.5 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.3b_rr_bprmax_opt.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.3b] PASSED
    echo [T6.3b] PASSED >> %results_log%
) else (
    echo [T6.3b] FAILED
    echo [T6.3b] FAILED >> %results_log%
)
echo. >> %results_log%

REM T6.3c: BPR-max with larger capacity
echo [T6.3c] BPR-max with layers=256 (larger capacity)
echo [T6.3c] %date% %time% >> %results_log%

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps loss=bpr-max,layers=256,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.05,learning_rate=0.08,momentum=0.5,bpreg=2.0,elu_param=0.5 ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  -d cpu ^
  -s output_data/T6.3c_rr_bprmax_large.pt >> %results_log% 2>&1

if %errorlevel% equ 0 (
    echo [T6.3c] PASSED
    echo [T6.3c] PASSED >> %results_log%
) else (
    echo [T6.3c] FAILED
    echo [T6.3c] FAILED >> %results_log%
)
echo. >> %results_log%

:skip_rr_bprmax

echo.
echo ===================================================
echo  Phase 6 Tests Completed
echo  Results logged to: %results_log%
echo ===================================================
echo.

pause
exit /b
