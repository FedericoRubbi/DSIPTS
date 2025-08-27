## Copyright 2023 Yunhao Zhang and Junchi Yan (https://github.com/Thinklab-SJTU/Crossformer?tab=Apache-2.0-1-ov-file#readme)
## Code modified for align the notation and the batch generation
## extended to all present in crossformer folder



from torch import  nn
import torch

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from typing import List,Union
from einops import  repeat
from ..data_structure.utils import beauty_string
from .utils import  get_scope
from .crossformer.cross_encoder import Encoder
from .crossformer.cross_decoder import Decoder
from .crossformer.cross_embed import DSW_embedding
from .utils import Embedding_cat_variables
from math import ceil
  
  
#    self, past_channels, past_steps, future_steps, seg_len, win_size = 4,
#                factor=10, d_model=512, hidden_size = 1024, n_head=8, n_layer_encoder=3, 
#                dropout=0.0, baseline = False,
  
class CrossFormer(Base):
    handle_multivariate = True
    handle_future_covariates = False
    handle_categorical_variables = False
    handle_quantile_loss = False

    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 d_model:int,
                 hidden_size:int,
                 n_head:int,
                 seg_len:int,
                 n_layer_encoder:int,
                 win_size:int,
                 factor:int=5,
                 remove_last = False,
                 dropout_rate:float=0.1,

                 **kwargs)->None:
        """CroosFormer (https://openreview.net/forum?id=vSVLM2j9eie)

        Args:
            d_model (int):  dimension of the attention model
            
     
            hidden_size (int): hidden size of the linear block
            n_head (int): number of heads
            seg_len (int): segment length (L_seg) see the paper for more details
            n_layer_encoder (int):  layers to use in the encoder
            win_size (int): window size for segment merg
            factor (int): num of routers in Cross-Dimension Stage of TSA (c) see the paper
            remove_last (boolean,optional): if true the model try to predic the difference respect the last observation.
            dropout_rate (float, optional):  dropout rate in Dropout layers. Defaults to 0.1.
        """
      
   
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        self.remove_last = remove_last
        

        # The padding operation to handle invisible sgemnet length
        self.pad_past_steps = ceil(1.0 *self.past_steps / seg_len) * seg_len
        self.pad_future_steps = ceil(1.0 * self.future_steps / seg_len) * seg_len
        self.past_steps_add = self.pad_past_steps - self.past_steps

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.past_channels, (self.pad_past_steps // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        ## Custom embeddings ##these are not used in crossformer
        #self.emb_past = Embedding_cat_variables(self.past_steps,emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        #self.emb_fut = Embedding_cat_variables(self.future_steps,emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
        #emb_past_out_channel = self.emb_past.output_channels
        #emb_fut_out_channel = self.emb_fut.output_channels

        self.encoder = Encoder(n_layer_encoder, win_size, d_model, n_head, hidden_size, block_depth = 1, \
                                    dropout = dropout_rate,in_seg_num = (self.pad_past_steps // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.past_channels, (self.pad_future_steps // seg_len), d_model))
        self.decoder = Decoder(seg_len, n_layer_encoder + 1, d_model, n_head, hidden_size, dropout_rate, \
                                    out_seg_num = (self.pad_future_steps // seg_len), factor = factor)
        
    def forward(self, batch):

        idx_target = batch['idx_target'][0]
        x_seq = batch['x_num_past'].to(self.device)#[:,:,idx_target]

        ## TODO add categorical to crossformer
        #if 'x_cat_past' in batch.keys():
        #    emb_past = self.emb_past(batch['x_cat_past'])
        #if 'x_cat_fut' in batch.keys():
        #    emb_fut = self.emb_fut(batch['x_cat_fut'])
                
        if self.remove_last:
            x_start = x_seq[:,-1,:].unsqueeze(1)
            x_seq[:,:,:]-=x_start   
 
        batch_size = x_seq.shape[0]
        if (self.past_steps_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.past_steps_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        
        

            
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)



        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        res = predict_y[:, :self.future_steps,:].unsqueeze(3)
        if self.remove_last:
            res+=x_start.unsqueeze(1)

        return res[:, :,idx_target,:]