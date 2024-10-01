import torch
import torch.nn as nn
import torch.nn.functional as F

class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout=0.3):
        super(ASRModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the output size of convolution layers dynamically
        # Assuming input_dim is the number of Mel features or spectrogram features
        self.conv_output_size = self._get_conv_output_size(input_dim)
        
        # Recurrent LSTM layers
        self.lstm = nn.LSTM(input_size=self.conv_output_size, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional doubles hidden_dim

    def _get_conv_output_size(self, input_dim):
        # Dummy forward pass through the convolutional layers to get the output size
        dummy_input = torch.zeros(1, 1, input_dim, 100)  # (batch_size, channels, freq, time)
        with torch.no_grad():
            conv_out = self.pool(torch.relu(self.conv1(dummy_input)))
            conv_out = self.pool(torch.relu(self.conv2(conv_out)))
        batch_size, channels, height, width = conv_out.size()
        return channels * height  # This will be used as input size for LSTM
    
    def forward(self, x):
        # Pass through convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Reshape for LSTM
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height * channels, width).transpose(1, 2)  # (batch, time, features)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Pass through the final fully connected layer
        x = self.fc(x[:, -1, :])  # Use the last time step's output for classification

        return x
