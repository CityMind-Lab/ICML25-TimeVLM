import torch
import torch.nn as nn

class QueryTimeSeriesInteraction(nn.Module):
    """
    Query-Time Series Interaction module for multimodal fusion.
    """
    def __init__(self, num_queries, time_series_embedding_dim, query_embedding_dim, hidden_dim, num_heads):
        super(QueryTimeSeriesInteraction, self).__init__()
        
        self.num_queries = num_queries
        self.time_series_embedding_dim = time_series_embedding_dim
        self.query_embedding_dim = query_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable query vectors, shape [num_queries, query_embedding_dim]
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, query_embedding_dim))
        
        # A linear layer to encode time-series embeddings into query_embedding_dim
        self.time_series_encoder = nn.Linear(time_series_embedding_dim, query_embedding_dim)

        # Multi-head attention for query-time series interaction
        self.multihead_attention = nn.MultiheadAttention(query_embedding_dim, num_heads, batch_first=True)

        # A linear layer to further transform the pooled result into a text-like hidden dimension
        self.text_vector_generator = nn.Linear(query_embedding_dim, hidden_dim)

    def forward(self, x_enc, patch_embedding):
        """
        Forward pass for the QueryTimeSeriesInteraction.

        Args:
            x_enc (torch.Tensor): shape [batch_size, seq_len, n_vars].
            patch_embedding (torch.Tensor): shape [[batch_size*n_vars, num_patches, d_model].

        Returns:
            torch.Tensor: A text-like vector representation of shape [batch_size, hidden_dim].
        """
        B, L, D = x_enc.shape
        time_series_embeddings = patch_embedding.view(B, D, -1, patch_embedding.shape[-1])  # [batch_size, n_vars, num_patches, d_model]
        time_series_embeddings = time_series_embeddings.view(B, -1, time_series_embeddings.shape[-1])   # [batch_size, (n_vars * num_patches), d_model]

        # Encode the time-series to match the query dimension: [batch_size, (n_vars * num_patches), query_dim]
        encoded_time_series = self.time_series_encoder(time_series_embeddings)

        # Expand the learnable query vectors for each batch, resulting in [batch_size, num_queries, query_dim]
        queries = self.query_embeddings.unsqueeze(0).repeat(B, 1, 1)

        # Apply multi-head attention, shape: [batch_size, num_queries, query_dim]
        interaction_output, _ = self.multihead_attention(queries, encoded_time_series, encoded_time_series)

        # Generate the final text-like vector, shape: [batch_size, num_queries, hidden_dim]
        text_vectors = self.text_vector_generator(interaction_output)
                        
        return text_vectors