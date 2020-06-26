class Config:
    num_layer = 2
    learning_rate = 1e-3
    use_gpu = True
    epoch = 50
    batch_size = 128
    max_gen_len = 200
    interval = 150
    embedding_dim = 128
    hidden_dim = 256
    model_prefix = 'checkpoint/tang'
    data_path = 'data/tang.npz'
    model_path = None
