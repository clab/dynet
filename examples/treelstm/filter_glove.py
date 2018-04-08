import codecs
import re
import os

data_dir = 'trees'
datasets = ['train', 'dev', 'test']
glove_origin_path = 'glove.840B.300d.txt'
glove_filtered_path = 'glove_filtered.txt'


def get_vocab(file_path):
    vocab = set()
    tokker = re.compile(r'([^ ()]+)\)')
    with codecs.open(file_path) as f:
        for line in f:
            for match in tokker.finditer(line.strip()):
                vocab.add(match.group(1))
    return vocab


vocab = set()
for dataset in datasets:
    tem_set = get_vocab(os.path.join(data_dir, dataset + '.txt'))
    vocab.update(tem_set)

total = cnt = 0
with codecs.open(glove_origin_path) as fin:
    with codecs.open(glove_filtered_path, 'w') as fout:
        for line in fin:
            total += 1
            word = line.split(' ', 1)[0]
            if word in vocab or word == '(' or word == ')':
                cnt += 1
                fout.write(line)
print('total: {}, after filtering: {}'.format(total, cnt))

