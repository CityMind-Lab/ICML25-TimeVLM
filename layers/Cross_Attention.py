import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, query, key, value):
        # query: [B, seq_len_q, hidden_dim]
        # key: [B, seq_len_k, hidden_dim]
        # value: [B, seq_len_k, hidden_dim]
        query = query.transpose(0, 1)  # [seq_len_q, B, hidden_dim]
        key = key.transpose(0, 1)      # [seq_len_k, B, hidden_dim]
        value = value.transpose(0, 1)  # [seq_len_k, B, hidden_dim]
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)  # [B, seq_len_q, hidden_dim]
        return attn_output