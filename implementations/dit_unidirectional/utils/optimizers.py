import torch
import torch.nn as nn
from utils.lr_schedulars import LRSched

class Optimizers:
    @staticmethod
    def get_optimizer(model, args):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        return opt
    
    @staticmethod
    def get_scheduler(opt, args):
        scheduler = LRSched.warmup_cosine(opt, args)
        return scheduler