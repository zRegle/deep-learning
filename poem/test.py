import numpy as np
from generate import *
from model import Poem
from config import Config

datas = np.load(Config.data_path, allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
model = Poem(len(ix2word), Config.embedding_dim, Config.hidden_dim)
map_location = 'cuda' if torch.cuda.is_available() else 'cuda'
model.load_state_dict(torch.load(Config.model_path, map_location=map_location))
if Config.use_gpu:
    model = model.to('cuda')

print('请输入生成模式:\n1.普通模式\n2.藏头诗模式')
mode = int(input())
print('请输入首句:')
start_words = str(input())
print('如有相关意境请输入，若无按回车继续:')
prefix_words = str(input())
gen = generate if mode == 1 else gen_acrostic
start_words = start_words.replace(',', u'，').replace('.', u'。').replace('?', u'？').replace('!', u'！')
prefix_words = prefix_words.replace(',', u'，').replace('.', u'。').replace('?', u'？').replace('!', u'！')
poetry = ''.join(gen(model, start_words, ix2word, word2ix, prefix_words))
print(poetry)
