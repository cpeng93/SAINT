import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
print(args)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    print('number of parameters:', count_parameters(model))    
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

