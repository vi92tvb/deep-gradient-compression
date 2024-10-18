from torchpack.mtpack.models.vision.cnn import CNNModel

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(CNNModel)
configs.model.num_classes = configs.dataset.num_classes
