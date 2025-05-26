import torch

def generate_multi_channel_time_series_prompt(self, x_enc, description, pred_len, seq_len, top_k=5):
    """
    Generate text prompts for the language model based on time series data.
    Each variable in the time series will have its own prompt.

    Args:
        x_enc (torch.Tensor): Input time series data, shape of [batch_size, seq_len, n_vars].
        description (str): Description of the dataset.
        pred_len (int): Prediction length.
        seq_len (int): Sequence length.
        top_k (int): Number of top variables to highlight (optional).

    Returns:
        List[str]: A list of prompts for each variable in each batch, with length B * n_vars.
    """
    B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

    # Initialize a list to store prompts for each variable in each batch
    prompts = []
    
    # Calculate statistics for each batch and each variable
    for b in range(B):
        for var in range(n_vars):
            # Extract the time series for the current variable
            var_series = x_enc[b, :, var]  # [seq_len]

            # Calculate statistics for the current variable
            min_value = torch.min(var_series).item()  # Minimum value
            max_value = torch.max(var_series).item()  # Maximum value
            median_value = torch.median(var_series).item()  # Median value
            trend = var_series.diff(dim=0).sum().item()  # Trend

            # Determine the trend direction
            trend_direction = "upward" if trend > 0 else "downward"

            # Generate prompt for the current variable
            prompt_parts = [
                f"The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_manager.max_input_text_length] if len(prompt) > self.vlm_manager.max_input_text_length else prompt
            prompts.append(prompt)

    return prompts

def generate_simple_multi_channel_time_series_prompt(self, x_enc, description, pred_len, seq_len, top_k=5):
    """
    Generate text prompts for the language model based on time series data.
    Each variable in the time series will have the same fixed prompt.

    Args:
        x_enc (torch.Tensor): Input time series data, shape of [batch_size, seq_len, n_vars].
        description (str): Description of the dataset.
        pred_len (int): Prediction length.
        seq_len (int): Sequence length.
        top_k (int): Number of top variables to highlight (optional).

    Returns:
        List[str]: A list of fixed prompts for each variable in each batch, with length B * n_vars.
    """
    B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

    # Fixed prompt content
    fixed_prompt = "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting."

    # Ensure the prompt does not exceed the maximum input text length
    fixed_prompt = fixed_prompt[:self.vlm_manager.max_input_text_length] if len(fixed_prompt) > self.vlm_manager.max_input_text_length else fixed_prompt

    # Repeat the fixed prompt for each variable in each batch
    prompts = [fixed_prompt] * (B * n_vars)

    return prompts
