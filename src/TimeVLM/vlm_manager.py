import sys
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers import Blip2Processor, Blip2Model

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.TimeVLM.vlm_custom import CustomVLM
from layers.models_mae import *
from transformers.models.vilt import *

class VLMManager:
    """
    Manager class to handle different VLM types (CLIP, BLIP2, ViLT).
    """
    def __init__(self, config):
        self.config = config
        self.vlm_type = config.vlm_type.lower()
        self.device = self._acquire_device()
        self._init_vlm()
        
    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vlm(self):
        if self.vlm_type == "clip":
            self._init_clip()
        elif self.vlm_type == "blip2":
            self._init_blip2()
        elif self.vlm_type == "vilt":
            self._init_vilt()
        elif self.vlm_type == "custom":
            self._init_custom()
        else:
            raise ValueError(f"Unsupported vlm_type: {self.vlm_type}. Choose from ['clip', 'blip2', 'vilt'].")
        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("VLM Learnable model parameters: {:,}".format(learnable_params))

    def _init_clip(self):
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
        self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = 9
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1), 
            nn.Sigmoid()
        ).to(self.device)

    def _init_blip2(self):
        BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
        self.processor = Blip2Processor.from_pretrained(BLIP_ARCH)
        self.model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 2560
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 1024
        self.fused_feature_len = 298

    def _init_vilt(self):
        VILT_ARCH = "dandelin/vilt-b32-finetuned-coco"
        self.processor = ViltProcessor.from_pretrained(VILT_ARCH)
        self.model = ViltModel.from_pretrained(VILT_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 768
        if self.config.w_out_query:
            self.fusion_dim = self.hidden_size
        else:
            self.fusion_dim = self.hidden_size
        self.max_input_text_length = 40
        self.fused_feature_len = 156
        
    def _init_custom(self):
        """
        Initialize the custom VLM.
        """
        self.model = CustomVLM(self.config)
        self.hidden_size = self.model.hidden_size
        self.max_input_text_length = 512  # Adjust based on text encoder

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value
        for child in model.children():
            self._set_requires_grad(child, value)

    def process_inputs(self, B, images, prompts):
        try: 
            if self.vlm_type == "clip":
                return self._process_clip_inputs(B, images, prompts)
            elif self.vlm_type == "blip2":
                return self._process_blip2_inputs(B, images, prompts)
            elif self.vlm_type == "vilt":
                return self._process_vilt_inputs(B, images, prompts)
            elif self.vlm_type == "custom":
                return self._process_custom_inputs(B, images, prompts)
        except Exception as e:
            print(f"Error processing inputs: {e}")
            print(f"Images shape: {images.shape}")
            print(f"Prompts: {prompts}")
            raise e

    def _process_clip_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        return image_features, text_features  # Both shape: [B, hidden_size]

    def _process_blip2_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True).language_model_outputs.hidden_states[-1]   # [B, seq_len, hidden_size]
        
        # Get text token count and extract features
        text_token_count = encoding["input_ids"].shape[1]
        text_features = outputs[:, :text_token_count, :]  # [B, text_token_count, hidden_size]
        image_features = outputs[:, text_token_count:, :]  # [B, seq_len - text_token_count, hidden_size]
        
        # Average pooling for both features
        text_features = text_features.mean(dim=1)  # [B, hidden_size]
        image_features = image_features.mean(dim=1)  # [B, hidden_size]
        
        return image_features, text_features  # Both shape: [B, hidden_size]
    
    def _process_vilt_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state  # Shape: [B, seq_len, hidden_size]
        
        # Get text token count and extract features
        text_token_count = encoding["input_ids"].shape[1]
        text_features = last_hidden_state[:, :text_token_count, :]  # [B, text_token_count, hidden_size]
        image_features = last_hidden_state[:, text_token_count:, :]  # [B, seq_len - text_token_count, hidden_size]
        
        # Average pooling for both features
        text_features = text_features.mean(dim=1)  # [B, hidden_size]
        image_features = image_features.mean(dim=1)  # [B, hidden_size]
        
        return image_features, text_features  # Both shape: [B, hidden_size]
    
    def _process_custom_inputs(self, B, images, prompts):
        vision_embeddings = self.model.get_vision_embeddings(images)    # Shape: [B, hidden_size]
        text_embeddings = self.model.get_text_embeddings(prompts)        # Shape: [B, hidden_size]
        return vision_embeddings, text_embeddings  # Both shape: [B, hidden_size]