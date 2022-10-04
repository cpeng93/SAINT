import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()
        # for param in self.model.SFENet1.parameters():
        #     param.requires_grad = False
        # for param in self.model.SFENet2.parameters():
        #     param.requires_grad = False
        # for param in self.model.RDBs.parameters():
        #     param.requires_grad = False
        # for param in self.model.GFF.parameters():
        #     param.requires_grad = False
        # for param in self.model.GW.parameters():
        #     param.requires_grad = False
        # for param in self.model.GFF_ref.parameters():
        #     param.requires_grad = False
        for param in self.model.RFN.parameters():
            param.requires_grad = False
        if not args.cpu and args.n_GPUs >= 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, dist):
        return self.model(x, dist)

    def get_model(self):
        return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                )
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs), strict=True
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                )
            )


    def test(self,inp,dist,factor):
        return self.model.module.test(inp,dist,factor)