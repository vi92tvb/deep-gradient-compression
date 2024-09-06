from torchpack.mtpack.utils.config import Config, configs
from torchpack.mtpack.models.vision.lstm import LSTM

# LSTM model config
configs.model = Config(LSTM)
configs.model.vocab_size = 3982  # Số lượng từ trong từ điển
configs.model.embedding_size = 256  # Kích thước của vectơ embedding
configs.model.hidden_size = 150  # Số lượng đơn vị ẩn trong mỗi tầng LSTM
configs.model.n_layers = 2  # Số lượng tầng LSTM
configs.model.output_size = 3
configs.model.dropout=0.25
