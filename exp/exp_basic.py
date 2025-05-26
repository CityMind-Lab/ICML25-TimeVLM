import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer,TimeLLM,VisionTS

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeLLM': TimeLLM,
            'VisionTS': VisionTS,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Stationary': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer
        }
        if args.model == 'TimeVLM':
            from src.TimeVLM import model as TimeVLM
            self.model_dict['TimeVLM'] = TimeVLM

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        if args.is_training:
            self._log_model_parameters()
        
        
    def _log_model_parameters(self):
        """
        打印模型参数。
        """
        def count_learnable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def count_total_parameters(model):
            return sum(p.numel() for p in model.parameters())

        learable_params = count_learnable_parameters(self.model)
        total_params = count_total_parameters(self.model)
        print(f"Learnable model parameters: {learable_params:,}")
        print(f"Total model parameters: {total_params:,}")
        

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
