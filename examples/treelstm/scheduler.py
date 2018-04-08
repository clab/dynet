import time
import dynet as dy
import numpy as np
from utils import acc_eval


class Scheduler:
    def __init__(self, model, train, dev, params):
        self.train, self.dev = train, dev
        self.model = model
        self.params = params

        self.trainer_param = getattr(dy, params['trainer'])(model.pc_param)
        self.trainer_embed = getattr(dy, params['trainer'])(model.pc_embed)
        self.trainer_param.learning_rate = params['learning_rate_param']
        self.trainer_embed.learning_rate = params['learning_rate_embed']
        all_trainers = [self.trainer_param, self.trainer_embed]
        for trainer in all_trainers:
            trainer.set_clip_threshold(-1)
            trainer.set_sparse_updates(params['sparse'])

    def exec_train(self, max_turns=1000):
        time_stamp = time.time()
        total_time = []
        best_acc = 0
        n_endure, endure_upper = 0, 10
        model_meta_file = None
        for i in range(max_turns):
            self.train.reset()
            time_start = time.time()
            for j, trees in enumerate(self.train.batches(batch_size=self.params['batch_size']), 1):
                dy.renew_cg()
                loss = self.model.losses_for_tree_batch(trees)
                loss += self.model.regularization_loss(coef=self.params['regularization_strength'])
                loss_value = loss.value()
                loss.backward()
                self.trainer_param.update()
                self.trainer_embed.update()

                if j % 50 == 0:
                    self.trainer_param.status()
                    print(loss_value)
            time_epoch = time.time() - time_start
            total_time.append(time_epoch)
            print('epoch {} time {}'.format(i, time_epoch))
            self.trainer_param.learning_rate *= self.params['learning_rate_decay']
            self.trainer_embed.learning_rate *= self.params['learning_rate_decay']

            acc = acc_eval(self.dev, self.model)
            best_acc, updated = max(acc, best_acc), acc > best_acc
            print("dev_acc=%.4f best_dev_acc=%.4f" % (acc, best_acc))

            if updated:
                self.model.delete(model_meta_file)
                model_meta_file = self.model.save(self.params['save_dir'], str(time_stamp) + '_' + str(i))
                n_endure = 0
            else:
                n_endure += 1
                if n_endure > endure_upper:
                    break
        self._print_time_statistics(total_time)
        return best_acc, model_meta_file

    @staticmethod
    def _print_time_statistics(total_time):
        print("N_EPOCH {}, MEAN {} s, STD {} s".format(len(total_time) - 1, np.mean(total_time[1:]), np.std(total_time[1:])))
