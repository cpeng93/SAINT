import os
import math
from decimal import Decimal

import utility
import scipy.misc
import torch
from torch.autograd import Variable
from tqdm import tqdm
import pickle
import numpy as np
import copy
import imageio.core.util
from medpy.io import load, save
from skimage.measure import compare_psnr

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device= torch.device('cuda')
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, dist, names, idx_scale) in enumerate(self.loader_train):
            lr, hr,dist = self.prepare(lr, hr, dist)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, dist) #should be 4 by 64 by 64 by 64
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {} Loss orig: {})'.format(
                    batch + 1, loss.item(), loss_orig.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc_ref = 0
                eval_acc_sag = 0
                eval_acc_cor = 0
                total_slice = 0
                # eval_acc = 0
                # eval_acc_orig = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, dist, filename, _) in enumerate(tqdm_test):

                    filename = filename[0]
                    lr, dist = self.prepare(lr, dist)
                    if self.args.stage == 1:
                        sag,cor = self.model.test(lr, dist,scale)
                        acc = utility.calc_psnr([sag, cor], hr, scale)
                        eval_acc_sag += acc[0]*sag.shape[-1]
                        eval_acc_cor += acc[1]*sag.shape[-1]
                        total_slice += sag.shape[-1]                                            
                        if self.args.save_results:
                            self.ckp.save_results(filename, [sag, cor], scale)

                
                    else:
                        ref,sag,cor = self.model.test(lr, dist,scale)
                        acc = utility.calc_psnr([sag, cor, ref], hr, scale)
                        eval_acc_sag += acc[0]*ref.shape[-1]
                        eval_acc_cor += acc[1]*ref.shape[-1]
                        eval_acc_ref += acc[2]*ref.shape[-1]
                        total_slice += sag.shape[-1]                                            
                        if self.args.save_results:
                            self.ckp.save_results(filename, [sag, cor, ref], scale)

                eval_acc_sag /= total_slice
                eval_acc_cor /= total_slice  
                if self.args.stage==2:
                    eval_acc_ref /= total_slice
                    eval_stat = eval_acc_ref
                    print('AVG, Final:',np.round(eval_acc_ref,2), 'MSR:',np.round(eval_acc_sag,2), np.round(eval_acc_cor,2))
                else:
                    eval_stat = (eval_acc_sag+eval_acc_cor)/2 
                    print('AVG MSR:',np.round(eval_acc_sag,2), np.round(eval_acc_cor,2))

                self.ckp.log[-1, idx_scale] = eval_stat
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))



    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

