@echo off
REM filepath: c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\run_winning_yoochoose.bat
REM Run GRU4Rec with Yoochoose Winning Parameters
REM Dataset: Yoochoose (full training set)
REM Loss: cross-entropy
REM Parameters: layers=480, batch_size=48, learning_rate=0.07, etc.

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
echo =========================================
echo  YOOCHOOSE - WINNING PARAMETERS TEST
echo =========================================
echo.
echo Configuration:
echo   Dataset: Yoochoose (full)
echo   Loss: cross-entropy
echo   Layers: 480
echo   Batch Size: 48
echo   Learning Rate: 0.07
echo   Epochs: 10
echo   Metrics: Recall@1, @5, @10, @20
echo.
echo Training model...
echo.

python run.py input_data/yoochoose-data/yoochoose_train_full.dat ^
  -ps "loss=cross-entropy,constrained_embedding=True,embedding=0,elu_param=0,layers=480,n_epochs=10,batch_size=48,dropout_p_embed=0.0,dropout_p_hidden=0.2,learning_rate=0.07,momentum=0.0,n_sample=2048,sample_alpha=0.2,bpreg=0.0,logq=1.0" ^
  -t input_data/yoochoose-data/yoochoose_test.dat ^
  -m 1 5 10 20 ^
  --eval-metrics recall_mrr,coverage,ild,diversity ^
  -d cpu ^
  -s output_data/yoochoose_xe_winning_final.pt

echo.
echo =========================================
echo  TRAINING COMPLETE
echo =========================================
echo Results saved to: output_data/yoochoose_xe_winning_final.pt
echo.

exit /b
