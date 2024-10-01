from torchpack.mtpack.utils.config import Config, configs
from torchpack.mtpack.models.vision.asr import ASRModel

# LSTM model config
configs.model = Config(ASRModel)
configs.model.input_dim = 13 # n_mfcc
configs.model.hidden_dim = 800
configs.model.output_dim = 2  # Genders
configs.model.num_layers = 5  # LSTM Layer
configs.model.dropout = 0.3  # Dropout
