from collections import OrderedDict

gru4rec_params = OrderedDict([
    ('loss', 'cross-entropy'),
    ('constrained_embedding', True),
    ('embedding', 0),
    ('elu_param', 0.0),
    ('layers', [112]),
    ('n_epochs', 4),
    ('batch_size', 128),
    ('dropout_p_embed', 0.0),
    ('dropout_p_hidden', 0.1),
    ('learning_rate', 0.08),
    ('momentum', 0.0),
    ('n_sample', 2048),
    ('sample_alpha', 0.2),
    ('bpreg', 0.0),
    ('logq', 1.0),
])