@echo off
REM filepath: c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\run_tests.bat
REM Comprehensive test suite for GRU4Rec - Execute from gru4rec_torch directory

cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
echo =========================================
echo  GRU4Rec Comprehensive Test Suite
echo  Version: 1.0
echo  Date: %date% %time%
echo =========================================
echo.

REM Create output directory if it doesn't exist
if not exist output_data mkdir output_data

REM Create results log
set results_log=output_data\test_results.log
echo Test Execution Log - %date% %time% > %results_log%
echo. >> %results_log%

REM PHASE 1: FOUNDATION VALIDATION
echo.
echo [PHASE 1] FOUNDATION VALIDATION
echo ================================
echo Phase 1: Foundation Validation >> %results_log%

call :run_test "T1.1" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=128,batch_size=64,n_epochs=1,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.1" "T1.1_sanity" %results_log%

call :run_test "T1.2" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=192,batch_size=64,n_epochs=5,n_sample=512,dropout_p_hidden=0.1,learning_rate=0.07" "T1.2_quick" %results_log%

REM PHASE 2: CONFIGURATION EXPLORATION
echo.
echo [PHASE 2] CONFIGURATION EXPLORATION
echo ===================================
echo Phase 2: Configuration Exploration >> %results_log%

REM Loss Function Ablation
echo.
echo [PHASE 2.1] Loss Function Ablation
call :run_test "T2.1a" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=128,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.07" "T2.1a_xe" %results_log%

call :run_test "T2.1b" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=bpr-max,layers=200,batch_size=64,n_epochs=5,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.05,momentum=0.4,bpreg=1.95,elu_param=0.5" "T2.1b_bprmax" %results_log%

REM Architecture Sweep
echo.
echo [PHASE 2.2] Architecture Sweep
call :run_test "T2.2_96" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=96,batch_size=128,n_epochs=6,n_sample=256,dropout_p_hidden=0.0,learning_rate=0.1" "T2.2_layers96" %results_log%

call :run_test "T2.2_112" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=112,batch_size=192,n_epochs=6,n_sample=256,dropout_p_hidden=0.15,learning_rate=0.085" "T2.2_layers112" %results_log%

call :run_test "T2.2_192" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=192,batch_size=64,n_epochs=6,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.07" "T2.2_layers192" %results_log%

REM Batch Size Impact
echo.
echo [PHASE 2.3] Batch Size Impact
call :run_test "T2.3_bs128" "input_data/yoochoose-data/yoochoose_train_valid.dat" "loss=cross-entropy,layers=112,batch_size=128,n_epochs=5,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08" "T2.3_batch128" %results_log%

REM PHASE 3: DATASET VALIDATION
echo.
echo [PHASE 3] DATASET VALIDATION
echo =============================
echo Phase 3: Dataset Validation >> %results_log%

call :run_test "T3.1" "input_data/yoochoose-data/yoochoose_train_full.dat" "loss=cross-entropy,layers=96,batch_size=128,n_epochs=2,n_sample=256,dropout_p_hidden=0.1,learning_rate=0.08" "T3.1_full" %results_log%

echo.
echo =========================================
echo  ALL TESTS COMPLETED
echo  Results logged to: %results_log%
echo =========================================
echo.

pause
exit /b

REM ============================================
REM Function to run a single test
REM ============================================
:run_test
setlocal
set test_name=%~1
set train_data=%~2
set params=%~3
set output_name=%~4
set log_file=%~5

echo.
echo [%test_name%] Starting...
echo [%test_name%] %date% %time% >> %log_file%

python run.py %train_data% ^
  -ps %params% ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 5 10 20 ^
  -d cpu ^
  -s output_data/%output_name%.pt

if %errorlevel% equ 0 (
    echo [%test_name%] ✅ PASSED >> %log_file%
    echo [%test_name%] ✅ PASSED
) else (
    echo [%test_name%] ❌ FAILED (exit code: %errorlevel%) >> %log_file%
    echo [%test_name%] ❌ FAILED
)
echo. >> %log_file%

endlocal
exit /b