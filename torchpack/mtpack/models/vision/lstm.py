import torch
import torch.nn as nn

__all__ = ['LSTM']

class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size=3, hidden_size=128, embedding_size=256, n_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Adjust output size according to the concatenated hidden states
        self.fc = nn.Linear(hidden_size * 4, output_size)  # hidden_size * 4 if using last_hidden + last_cell + lstm_out

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)

        # Taking the last hidden state from the last layer
        last_hidden = hn[-1]  # From the last layer
        last_cell = cn[-1]    # From the last layer
        last_lstm_out = lstm_out[:, -1, :]  # The output from the last time step

        # Optionally include only the last hidden state or others as needed
        combined = torch.cat((last_hidden, last_cell, last_lstm_out), dim=1)

        out = self.dropout(combined)
        out = self.fc(out)

        return out
