# Run
python run.py /path/to/training_data_file -t /path/to/test_data_file -m 1 5 10 20 -ps layers=224,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0,loss=bpr-max,constrained_embedding=True,elu_param=0.5,n_epochs=10 -d cuda:0 -s /path/to/save_model.pt

python run.py yoochoose-clicks-reduced.tsv -t yoochoose-data\yoochoose-test.tsv -m 1 5 10 20 -ps layers=32,batch_size=3,dropout_p_embed=0.0,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0,n_sample=256,sample_alpha=1.0,bpreg=0.0,logq=0.0,loss=bpr-max,constrained_embedding=False,elu_param=0.0,n_epochs=3 -d cpu -s save_model_test.pt

python run.py yoochoose-clicks.tsv -t yoochoose-data\yoochoose-test.tsv -m 1 5 10 20 -ps layers=32,batch_size=3,dropout_p_embed=0.0,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0,n_sample=256,sample_alpha=1.0,bpreg=0.0,logq=0.0,loss=bpr-max,constrained_embedding=False,elu_param=0.0,n_epochs=3 -d cpu -s save_model.pt

# Save a model
python run.py /path/to/training_data_file -ps layers=128,batch_size=128,... -s /path/to/save_model.pt

# Load a model and evaluate
python run.py /path/to/save_model.pt -l -t /path/to/test_data_file -m 1 5 10 20 -e conservative -d cuda:0

python paropt.py /path/to/training_data /path/to/validation_data -pm mrr -m 20 -fm 1 5 10 20 -e conservative -fp n_sample=2048,logq=1.0,loss=cross-entropy,constrained_embedding=True,n_epochs=10 -d cuda:0 -opf /path/to/parameter_space.json -n 200

python run.py /path/to/training_data_file -pf /path/to/parameter_file.py -s /path/to/save_model.pt -d cuda:0

# Cross-Entropy Loss
python run.py /path/to/training_data_file -ps loss=cross-entropy,...

# BPR-Max Loss
python run.py /path/to/training_data_file -ps loss=bpr-max,...