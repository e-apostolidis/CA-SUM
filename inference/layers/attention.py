# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        """ The basic Attention 'cell' containing the learnable parameters of Q, K and V.

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.Wk = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wq = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wv = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.out = nn.Linear(in_features=output_size+2, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_entropy(logits):
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T].
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T].
        """
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        # https://stats.stackexchange.com/a/207093 Maximum value of entropy is log(k), where k the # of used categories.
        # Here k is when all the values of a row is different of each other (i.e., k = # of video frames)
        return _entropy / np.log(logits.shape[0])

    def forward(self, x):
        """ Compute the weighted frame features, through the Block diagonal sparse attention matrix and the estimates of
        the frames attentive uniqueness and the diversity.

        :param torch.Tensor x: Frame features with shape [T, input_size].
        :return: A tuple of:
                    y: The computed weighted features, with shape [T, input_size].
                    att_win : The Block diagonal sparse attention matrix, with shape [T, T].
        """
        # Compute the pairwise dissimilarity of each frame, on the initial feature space (GoogleNet features)
        x_unit = F.normalize(x, p=2, dim=1)
        similarity = x_unit @ x_unit.t()
        diversity = 1 - similarity

        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        energies = torch.matmul(Q, K.transpose(1, 0))
        att_weights = self.softmax(energies)

        # Entropy is a measure of uncertainty: Higher value means less information.
        entropy = self.get_entropy(logits=energies)
        entropy = F.normalize(entropy, p=1, dim=-1)

        # Compute the mask to form the Block diagonal sparse attention matrix
        D = self.block_size
        num_blocks = math.ceil(energies.shape[0] / D)
        keepingMask = torch.ones(num_blocks, D, D, device=att_weights.device)
        keepingMask = torch.block_diag(*keepingMask)[:att_weights.shape[0], :att_weights.shape[0]]
        zeroingMask = (1 - keepingMask)
        att_win = att_weights * keepingMask

        # Pick those frames that are "invisible" to a frame, aka outside the block (mask)
        attn_remainder = att_weights * zeroingMask
        div_remainder = diversity * zeroingMask

        # Compute non-local dependencies based on the diversity of those frames
        dep_factor = (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
        dep_factor = dep_factor.unsqueeze(0).expand(dep_factor.shape[0], -1)
        masked_dep_factor = dep_factor * keepingMask
        att_win += masked_dep_factor

        y = torch.matmul(att_win, V)
        characteristics = (entropy, dep_factor[0, :])
        characteristics = torch.stack(characteristics).detach()
        outputs = torch.cat(tensors=(y, characteristics.t()), dim=-1)

        y = self.out(outputs)
        return y, att_win.clone()


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = SelfAttention(input_size=256, output_size=128, block_size=30).cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
