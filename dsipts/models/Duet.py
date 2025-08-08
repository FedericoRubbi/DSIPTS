

import torch
import torch.nn as nn
import numpy as np

from .duet.layers import Linear_extractor_cluster
from .duet.masked import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
from einops import rearrange

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import QuantileLossMO,Permute, get_activation

from typing import List, Union
from ..data_structure.utils import beauty_string
from .utils import  get_scope




class Duet(Base):
    handle_multivariate = True
    handle_future_covariates = False # or at least it seems...
    handle_categorical_variables = True #solo nel encoder
    handle_quantile_loss = True # NOT EFFICIENTLY ADDED, TODO fix this
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 out_channels: int,
                 past_steps: int,
                 future_steps: int, 
                 past_channels: int,
                 future_channels: int,
                 embs: List[int],

                 # specific params
                 factor:int,
                 d_model: int,
                 n_head: int,
                 n_layer: int,
                 CI: int,
                 d_ff: int,
                 noisy_gating:bool,
                 num_experts: int,
                 kernel_size:int,
                 hidden_size:int,
                 k: int,
                 
                 dropout_rate: float=0.1,
                 activation: str='',
                 
                 persistence_weight:float=0.0,
                 loss_type: str='l1',
                 quantiles:List[float]=[],
                 optim:Union[str,None]=None,
                 optim_config:Union[dict,None]=None,
                 scheduler_config:Union[dict,None]=None,
                 **kwargs)->None:
        """

   
        Args:
         
        """
        
        super().__init__(**kwargs)
        if activation == 'torch.nn.SELU':
            beauty_string('SELU do not require BN','info',self.verbose)
            use_bn = False
        if isinstance(activation,str):
            activation = get_activation(activation)
        self.save_hyperparameters(logger=False)

        # self.dropout = dropout_rate
        self.persistence_weight = persistence_weight 
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type
        self.future_steps = future_steps
                
        if len(quantiles)==0:
            self.mul = 1
            self.use_quantiles = False
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()
        else:
            assert len(quantiles)==3, beauty_string('ONLY 3 quantiles premitted','info',True)
            self.mul = len(quantiles)
            self.use_quantiles = True
            self.loss = QuantileLossMO(quantiles)


        ##my update
        CAT = 0
        self.embs = nn.ModuleList()

        for j in embs:
            self.embs.append(nn.Embedding(j+1,d_model))
            CAT = 1 #TODO fix this


        self.cluster = Linear_extractor_cluster(noisy_gating,
                                                num_experts,
                                                past_steps,
                                                k,
                                                d_model,
                                                past_channels+CAT,
                                                CI,kernel_size,
                                                hidden_size)
        self.CI = CI
        self.n_vars = out_channels
        self.mask_generator = Mahalanobis_mask(future_steps)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            factor,
                            attention_dropout=dropout_rate,
                            output_attention=0,
                        ),
                        d_model,
                        n_head,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout_rate,
                    activation=activation,
                )
                for _ in range(n_layer)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.linear_head = nn.Sequential(nn.Linear(d_model, future_steps), nn.Dropout(dropout_rate))




    def forward(self, batch:dict)-> float:
        # x: [Batch, Input length, Channel]
        x_enc = batch['x_num_past'].to(self.device)
        idx_target = int(batch['idx_target'][0])
        BS = x_enc.shape[0]
        if 'x_cat_past' in batch.keys():
            x_mark_enc =  batch['x_cat_past'].to(self.device)
            tmp = []
            for i in range(len(self.embs)):
                tmp.append(self.embs[i](x_mark_enc[:,:,i]))
            x_mark_enc = torch.cat(tmp,2)
      
        x_enc = torch.concat([x_enc,x_mark_enc.sum(axis=2).unsqueeze(-1)],axis=-1)
        
        if self.CI:
            channel_independent_input = rearrange(x_enc, 'b l n -> (b n) l 1')

            reshaped_output, _ = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=x_enc.shape[0])

        else:
            temporal_feature, _ = self.cluster(x_enc)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            changed_input = rearrange(x_enc, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, _ = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")[:,:,idx_target]

        return output.reshape(BS,self.future_steps,self.n_vars,self.mul)



