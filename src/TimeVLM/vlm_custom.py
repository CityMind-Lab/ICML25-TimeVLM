import sys
import torch
import torch.nn as nn

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Cross_Attention import CrossAttention
from layers.models_mae import *
from transformers.models.vilt import *

class CustomVLM(nn.Module):
    """
    Custom Vision-Language Model that handles separate feature extraction for vision and text.
    """
    def __init__(self, config):
        super(CustomVLM, self).__init__()
        self.config = config
        self.device = self._acquire_device()
        
        # Initialize hidden_size
        self.hidden_size = 768  # Example hidden size, can be adjusted
        
        # Initialize vision and text encoders
        self._init_vision_encoder()
        self._init_text_encoder()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vision_encoder(self):
        """
        Initialize the vision encoder (e.g., ViT or ResNet).
        """
        from transformers import ViTModel, ViTFeatureExtractor
        self.vision_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder.to(self.device)
        self._set_requires_grad(self.vision_encoder, self.config.finetune_vlm)

    def _init_text_encoder(self):
        """
        Initialize the text encoder (e.g., BERT or RoBERTa).
        """
        from transformers import BertTokenizer, BertModel
        self.text_processor = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder.to(self.device)
        self._set_requires_grad(self.text_encoder, self.config.finetune_vlm)

    def _set_requires_grad(self, model, value):
        """
        Set requires_grad for all parameters in a model.
        """
        for param in model.parameters():
            param.requires_grad = value

    def get_vision_embeddings(self, images):
        """
        Extract vision embeddings from images.
        
        Args:
            images (List[PIL.Image]): List of input images.
        
        Returns:
            torch.Tensor: Vision embeddings of shape [B, hidden_size].
        """
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.vision_encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling over patches

    def get_text_embeddings(self, texts):
        """
        Extract text embeddings from texts.
        
        Args:
            texts (List[str]): List of input texts.
        
        Returns:
            torch.Tensor: Text embeddings of shape [B, hidden_size].
        """
        inputs = self.text_processor(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding