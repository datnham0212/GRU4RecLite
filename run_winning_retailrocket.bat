@echo off
REM filepath: c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\run_winning_retailrocket.bat
REM Run GRU4Rec with RetailRocket Winning Parameters
REM Dataset: RetailRocket (full training set)
REM Loss: bpr-max
REM Parameters: layers=224, batch_size=80, learning_rate=0.05, bpreg=1.95, etc.

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo.
echo =========================================
echo  RETAILROCKET - WINNING PARAMETERS TEST
echo =========================================
echo.
echo Configuration:
echo   Dataset: RetailRocket (full)
echo   Loss: bpr-max
echo   Layers: 224
echo   Batch Size: 80
echo   Learning Rate: 0.05
echo   BPR Regularization: 1.95
echo   Epochs: 10
echo   Metrics: Recall@1, @5, @10, @20
echo.
echo Training model...
echo.

python run.py input_data/retailrocket-data/retailrocket_train_full.dat ^
  -ps "loss=bpr-max,constrained_embedding=True,embedding=0,elu_param=0.5,layers=224,n_epochs=10,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0" ^
  -t input_data/retailrocket-data/retailrocket_test.dat ^
  -m 1 5 10 20 ^
  --eval-metrics recall_mrr,coverage,ild,diversity ^
  -d cpu ^
  -s output_data/retailrocket_bprmax_winning_final.pt

echo.
echo =========================================
echo  TRAINING COMPLETE
echo =========================================
echo Results saved to: output_data/retailrocket_bprmax_winning_final.pt
echo.

exit /b
