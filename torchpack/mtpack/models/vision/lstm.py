import torch
import torch.nn as nn

__all__ = ['LSTM']

class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=128, embedding_size=256, n_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 3, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)

        last_hidden = hn[-1]
        last_cell = cn[-1]
        last_lstm_out = lstm_out[:, -1, :]

        combined = torch.cat((last_hidden, last_cell, last_lstm_out), dim=1)
        
        # out = self.dropout(out)
        out = self.fc(combined)

        return out
