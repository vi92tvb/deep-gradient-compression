from torchvision.models import resnet18

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(resnet18)
configs.model.num_classes = configs.dataset.num_classes
