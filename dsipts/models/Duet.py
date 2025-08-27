
## Copyright 2025    DUET (https://github.com/decisionintelligence/DUET)
## Code modified for align the notation and the batch generation
## extended to all present in duet and autoformer folder

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
from .utils import Embedding_cat_variables




class Duet(Base):
    handle_multivariate = True
    handle_future_covariates = False # or at least it seems...
    handle_categorical_variables = True #solo nel encoder
    handle_quantile_loss = True # NOT EFFICIENTLY ADDED, TODO fix this
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 

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


        self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        #self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        emb_past_out_channel = self.emb_past.output_channels
        #emb_fut_out_channel = self.emb_fut.output_channels


       


        self.cluster = Linear_extractor_cluster(noisy_gating,
                                                num_experts,
                                                self.past_steps,
                                                k,
                                                d_model,
                                                self.past_channels+emb_past_out_channel,
                                                CI,kernel_size,
                                                hidden_size)
        self.CI = CI
        self.n_vars = self.out_channels
        self.mask_generator = Mahalanobis_mask(self.future_steps)
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

        self.linear_head = nn.Sequential(nn.Linear(d_model, self.future_steps), nn.Dropout(dropout_rate))




    def forward(self, batch:dict)-> float:
        # x: [Batch, Input length, Channel]
        x_enc = batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        BS = x_enc.shape[0]
        #if 'x_cat_future' in batch.keys():
        #    emb_fut = self.emb_fut(BS,batch['x_cat_future'].to(self.device))
        #else:
        #    emb_fut = self.emb_fut(BS,None)
        if 'x_cat_past' in batch.keys():
            emb_past = self.emb_past(BS,batch['x_cat_past'].to(self.device))
        else:
            emb_past = self.emb_past(BS,None)
   
      
        x_enc = torch.concat([x_enc,emb_past],axis=-1)
        
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



