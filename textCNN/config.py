class Config:
    update_w2v = True
    vocab_size = 59290
    classes = 2
    max_sen_len = 75
    batch_size = 128
    embedding_dim = 50
    epoch = 10
    lr = 1e-3
    drop_keep_prob = 0.5
    num_filters = 256
    kernel_sizes = [2, 3, 4]
    interval = 20
    save_path = 'checkpoints/'
    train_path = 'Dataset/train.txt'
    dev_path = 'Dataset/validation.txt'
    test_path = 'Dataset/test.txt'
    word2id_path = 'Dataset/word2id.txt'
    pre_word2vec_path = 'Dataset/wiki_word2vec_50.bin'
