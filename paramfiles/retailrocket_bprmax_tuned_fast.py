from collections import OrderedDict

gru4rec_params = OrderedDict([
	('loss', 'bpr-max'),
	('constrained_embedding', True),
	('embedding', 0),
	('elu_param', 0.5),
	('layers', [224]),
	('n_epochs', 3),  # tuned fast: fewer epochs
	('batch_size', 80),
	('dropout_p_embed', 0.2),  # lighter dropout for speed
	('dropout_p_hidden', 0.02),
	('learning_rate', 0.05),
	('momentum', 0.4),
	('n_sample', 2048),
	('sample_alpha', 0.4),
	('bpreg', 1.0),  # lighter regularization for speed
	('logq', 0.0),
])
