from torchpack.mtpack.utils.config import Config, configs
from torchpack.mtpack.models.vision.lstm import LSTM

# LSTM model config
configs.model = Config(LSTM)
configs.model.vocab_size = 1587509
configs.model.embedding_size = 100
configs.model.hidden_size = 1500
configs.model.n_layers = 2
configs.model.output_size = 3
configs.model.dropout=0.2
