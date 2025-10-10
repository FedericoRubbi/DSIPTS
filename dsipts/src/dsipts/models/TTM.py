import torch
import numpy as np
from torch import  nn

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base


from .ttm.utils import get_model, get_frequency_token, count_parameters


class TTM(Base):
    def __init__(self, 
                model_path:str,
                freq_prefix_tuning:bool,
                freq:str,
                prefer_l1_loss:bool,  # exog: set true to use l1 loss
                prefer_longer_context:bool,
                prediction_channel_indices,
                exogenous_channel_indices,
                decoder_mode,
                fcm_context_length,
                fcm_use_mixer,
                fcm_mix_layers,
                fcm_prepend_past,
                enable_forecast_channel_mixing,
                remove_last = False,
                **kwargs)->None:
        """TODO and FIX for future and past categorical variables
        
        Args:
        
        """
        super(TTM, self).__init__()
        self.save_hyperparameters(logger=False)
        self.remove_last = remove_last
        self.freq = freq
        self.extend_variables = False

        # NOTE: For Hydra
        prediction_channel_indices = list(prediction_channel_indices)
        exogenous_channel_indices = list(exogenous_channel_indices)

        if self.use_quantiles:
            self.extend_variables = True
            if self.out_channels * 3 != len(prediction_channel_indices):
                prediction_channel_indices, exogenous_channel_indices, num_input_channels = self.__add_quantile_features(prediction_channel_indices, 
                                                                                                                         exogenous_channel_indices, 
                                                                                                                         self.out_channels)
                self.past_channels = num_input_channels
        
        self.model = get_model(
            model_path=model_path,
            context_length=self.past_steps,
            prediction_length=self.future_steps,
            freq_prefix_tuning=freq_prefix_tuning,
            freq=freq,
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            num_input_channels=self.past_channels,
            decoder_mode=decoder_mode,
            prediction_channel_indices=prediction_channel_indices,
            exogenous_channel_indices=exogenous_channel_indices,
            fcm_context_length=fcm_context_length,
            fcm_use_mixer=fcm_use_mixer,
            fcm_mix_layers=fcm_mix_layers,
            fcm_prepend_past=fcm_prepend_past,
            enable_forecast_channel_mixing=enable_forecast_channel_mixing,
        )
        self.__freeze_backbone()

    def __add_quantile_features(self, prediction_channel_indices, exogenous_channel_indices, out_channels):
        prediction_channel_indices = list(range(out_channels * 3))
        exogenous_channel_indices = [prediction_channel_indices[-1] + i for i in range(1, len(exogenous_channel_indices)+1)]
        num_input_channels = len(prediction_channel_indices) + len(exogenous_channel_indices)
        return prediction_channel_indices, exogenous_channel_indices, num_input_channels

    def __freeze_backbone(self):
        """
        Freeze the backbone of the model.
        This is useful when you want to fine-tune only the head of the model.
        """
        print(
            "Number of params before freezing backbone",
            count_parameters(self.model),
        )
        # Freeze the backbone of the model
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        # Count params
        print(
            "Number of params after freezing the backbone",
            count_parameters(self.model),
        )
    
    def __scaler(self, input):
        #new_data = torch.tensor([MinMaxScaler().fit_transform(step_data) for step_data in data])
        for i, e in enumerate(self.embs):
            input[:,:,i] = input[:, :, i] / (e-1)
        return input
    
    def __build_tupla_indexes(self, size, target_idx, current_idx):
        permute = list(range(size))
        history = dict()
        for j, i in enumerate(target_idx):
            c = history.get(current_idx[j], current_idx[j])
            permute[i], permute[c] = current_idx[j], i
            history[i] = current_idx[j]


    def __permute_indexes(self, values, target_idx, current_idx):
        if current_idx is None or target_idx is None:
            raise ValueError("Indexes cannot be None")
        if sorted(current_idx) != sorted(target_idx):
            return values[..., self.__build_tupla_indexes(values.shape[-1], target_idx, current_idx)]
        return values
    
    def __extend_with_quantile_variables(self, x, original_indexes):
        covariate_indexes = [i for i in range(x.shape[-1]) if i not in original_indexes]
        covariate_tensors = x[..., covariate_indexes]

        new_tensors = [x[..., target_index] for target_index in original_indexes for _ in range(3)]

        new_original_indexes = list(range(len(original_indexes) * 3))
        return torch.cat([torch.stack(new_tensors, dim=-1), covariate_tensors], dim=-1), new_original_indexes
    
    def forward(self, batch):
        x_enc = batch['x_num_past']
        original_indexes = batch['idx_target'][0].tolist()
        original_indexes_future = batch['idx_target_future'][0].tolist()


        if self.extend_variables:
            x_enc, original_indexes = self.__extend_with_quantile_variables(x_enc, original_indexes)

        if 'x_cat_past' in batch.keys():
            x_mark_enc = batch['x_cat_past'].to(torch.float32).to(self.device)
            x_mark_enc = self.__scaler(x_mark_enc)
            past_values = torch.cat((x_enc,x_mark_enc), axis=-1).type(torch.float32)
        else:
            past_values = x_enc
        
        x_dec = torch.tensor([]).to(self.device)
        if 'x_num_future' in batch.keys(): 
            x_dec = batch['x_num_future'].to(self.device)
            if self.extend_variables:
                x_dec, original_indexes_future = self.__extend_with_quantile_variables(x_dec, original_indexes_future)
        if 'x_cat_future' in batch.keys():
            x_mark_dec = batch['x_cat_future'].to(torch.float32).to(self.device)
            x_mark_dec = self.__scaler(x_mark_dec)
            future_values = torch.cat((x_dec, x_mark_dec), axis=-1).type(torch.float32)
        else:
            future_values = x_dec

        if self.remove_last:
            idx_target = batch['idx_target'][0]
            x_start = x_enc[:,-1,idx_target].unsqueeze(1)
            x_enc[:,:,idx_target]-=x_start 


        past_values = self.__permute_indexes(past_values, self.model.prediction_channel_indices, original_indexes)

        
        future_values = self.__permute_indexes(future_values, self.model.prediction_channel_indices, original_indexes_future)

        freq_token = get_frequency_token(self.freq).repeat(x_enc.shape[0])

        res = self.model(
            past_values= past_values,
            future_values= future_values,
            past_observed_mask = None,
            future_observed_mask = None,
            output_hidden_states =  False,
            return_dict = False,
            freq_token= freq_token,
            static_categorical_values = None
        )
        #args = None
        #res = self.model(**args)
        BS = res.shape[0]
        return res.reshape(BS,self.future_steps,-1,self.mul)
        
    