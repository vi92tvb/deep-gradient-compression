import torch

from torchpack.mtpack.datasets.vision import VietnameseImage
from torchpack.mtpack.utils.config import Config, configs

# dataset
configs.dataset = Config(VietnameseImage)
configs.dataset.root = './data/vietnameseimage'
configs.dataset.num_classes = 30
configs.dataset.image_size = 224

# training
configs.train.num_epochs = 50
configs.train.batch_size = 32

# optimizer

configs.train.optimizer.lr = 0.1
configs.train.optimizer.weight_decay = 1e-4
configs.train.optimize_bn_separately = False

# scheduler
configs.train.scheduler = Config(torch.optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs - configs.train.warmup_lr_epochs
