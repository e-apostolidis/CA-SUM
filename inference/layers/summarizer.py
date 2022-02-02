# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.attention import SelfAttention


class CA_SUM(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        """ Class wrapping the CA-SUM model; its key modules and parameters.
        
        :param int input_size: The expected input feature size.
        :param int output_size: The produced output feature size.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(CA_SUM, self).__init__()

        self.attention = SelfAttention(input_size=input_size, output_size=output_size, block_size=block_size)
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features):
        """ Produce frame-level importance scores from the frame features, using the CA-SUM model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by 
        using the pool5 layer of GoogleNet.
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        residual = frame_features
        weighted_value, attn_weights = self.attention(frame_features)
        y = residual + weighted_value
        y = self.drop(y)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, attn_weights


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = CA_SUM(input_size=256, output_size=128, block_size=30).cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
