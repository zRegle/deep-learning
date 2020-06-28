import torch
from config import Config


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_word_len = len(start_words)
    _input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    if Config.use_gpu:
        _input = _input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(_input, hidden)
            _input = _input.data.new([word2ix[word]]).view(1, 1)

    for i in range(Config.max_gen_len):
        output, hidden = model(_input, hidden)
        if i < start_word_len:
            w = results[i]
            _input = _input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_idx = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_idx]
            results.append(w)
            _input = _input.data.new([top_idx]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    results = []
    start_word_len = len(start_words)
    _input = torch.Tensor([word2ix['<START>']]).view(1, 1,).long()
    if Config.use_gpu:
        _input = _input.cuda()
    hidden = None

    idx = 0
    pre_word = '<START>'

    if prefix_words:
        for w in prefix_words:
            output, hidden = model(_input, hidden)
            _input = _input.data.new([word2ix[w]]).view(1, 1)

    for i in range(Config.max_gen_len):
        output, hidden = model(_input, hidden)
        top_idx = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_idx]
        if pre_word in {u'。', u'！', '<START>'}:
            if idx == start_word_len:
                break
            else:
                w = start_words[idx]
                idx += 1
                _input = _input.data.new([word2ix[w]]).view(1, 1)
        else:
            _input = _input.data.new([word2ix[w]]).view(1, 1)
        results.append(w)
        pre_word = w
    return results
