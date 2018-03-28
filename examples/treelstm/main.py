from __future__ import print_function
import _dynet as dy
dyparams = dy.DynetParams()
dyparams.from_args()
dyparams.set_autobatch(True)
dyparams.init()

import time
import os
import argparse

from model import TreeLSTMClassifier
from utils import get_embeds, acc_eval
from scheduler import Scheduler
from dataloader import DataLoader
from gridsearch import GridSearch

data_dir = 'trees'
glove_path = 'glove_filtered.txt'


def establish_args():
    parser = argparse.ArgumentParser()
    # dynet global setting
    parser.add_argument("--dynet-seed", default=0, type=int)
    parser.add_argument("--dynet-mem", default=512, type=int)
    parser.add_argument("--dynet-gpus", default=0, type=int)

    # control parameters
    parser.add_argument('--mode', default='train', help='available modes: [train, search, test]')
    parser.add_argument('--model_meta_file', default=None, type=str)

    # scheduler parameters
    parser.add_argument('--trainer', default='AdagradTrainer', help='trainer name in dynet')
    parser.add_argument('--sparse', default=1, type=int, help='sparse update 0/1')
    parser.add_argument('--learning_rate_param', default=0.05, type=float)
    parser.add_argument('--learning_rate_embed', default=0.005, type=float)
    parser.add_argument('--save_dir', default='saved_models')
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--regularization_strength', default=1e-4, type=float)
    
    # model parameters
    parser.add_argument('--use_glove', default=False, action='store_true', help='Use glove vectors or not.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--wembed_size', default=300, type=int, help='embedding size')
    parser.add_argument('--hidden_size', default=150, type=int, help='hidden size')

    args = parser.parse_args()

    # check and ensure feasibility
    if args.use_glove and args.wembed_size != 300:
        print('Warning: word embedding size must be 300 when using glove, auto adjusted.')
        args.wembed_size = 300
    assert (args.mode in ['train', 'test', 'search']), 'Wrong mode, [train, test, search] available now'
    if args.mode == 'test':
        assert(args.model_meta_file is not None), "Model meta not specified in evaluation mode."
    else:
        meta_path = os.path.join(args.save_dir, 'meta')
        param_path = os.path.join(args.save_dir, 'param')
        embed_path = os.path.join(args.save_dir, 'embed')
        if not os.path.exists(meta_path): os.makedirs(meta_path)
        if not os.path.exists(param_path): os.makedirs(param_path)
        if not os.path.exists(embed_path): os.makedirs(embed_path)
    return args


start = time.time()
args = establish_args()

scheduler_params = {
    'trainer': args.trainer,
    'sparse': args.sparse == 1,
    'learning_rate_param': args.learning_rate_param,
    'learning_rate_embed': args.learning_rate_embed,
    'learning_rate_decay': 0.99,
    'save_dir': args.save_dir,
    'batch_size': args.batch_size,
    'regularization_strength': args.regularization_strength
}

model_params = {
    'wembed_size': args.wembed_size,
    'hidden_size': args.hidden_size,
    'dropout_rate': args.dropout_rate
}


train = DataLoader(os.path.join(data_dir, 'train.txt'))
dev = DataLoader(os.path.join(data_dir, 'dev.txt'))
test = DataLoader(os.path.join(data_dir, 'test.txt'))

word_embed, w2i = get_embeds(glove_path)
if not args.use_glove: word_embed = None

print("startup time: %r" % (time.time() - start))


def exec_train(model_params, scheduler_params):
    model = TreeLSTMClassifier(n_classes=5, w2i=w2i, word_embed=word_embed,
                               params=model_params)
    scheduler = Scheduler(model, train, dev, scheduler_params)
    return scheduler.exec_train()


if args.mode == 'train':
    acc, model_meta_file = exec_train(model_params, scheduler_params)
elif args.mode == 'search':
    search = GridSearch('grid_search.txt')
    f = open('grid_search_result.txt', 'w')
    best_acc, model_meta_file = 0, None
    for params in search:
        sp, mp = scheduler_params.copy(), model_params.copy()
        sp.update(params), mp.update(params)
        acc, model_meta_file_tmp = exec_train(mp, sp)
        f.write(str(params))
        f.write(str(acc))
        best_acc, updated = max(best_acc, acc), acc > best_acc
        if updated:
            model_meta_file = model_meta_file_tmp
        print('params {} acc {} best_acc {}'.format(params, acc, best_acc))
    f.close()
else:
    model_meta_file = args.model_meta_file


def eval_model(model_meta_file):
    print('model_meta_file {}'.format(model_meta_file))
    model = TreeLSTMClassifier(n_classes=5, w2i=w2i, word_embed=word_embed,
                               params=model_params, model_meta_file=model_meta_file)
    acc = acc_eval(test, model)
    print('test acc%.4f' % acc)


eval_model(model_meta_file)
