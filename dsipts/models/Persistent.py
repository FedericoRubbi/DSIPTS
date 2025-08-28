
from torch import nn

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
from .utils import L1Loss
from ..data_structure.utils import beauty_string
from .utils import  get_scope

class Persistent(Base):
    handle_multivariate = True
    handle_future_covariates = False
    handle_categorical_variables = False
    handle_quantile_loss = False
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    def __init__(self, 
                 **kwargs)->None:
 
        
    
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        #self.optim = None
        #self.loss = L1Loss()
        self.fake = nn.Linear(1,1)
        self.use_quantiles = False
        #self.loss_type = 'l1'
        #self.loss = nn.L1Loss()
        
    def forward(self, batch):
        """It is mandatory to implement this method

        Args:
            batch (dict): batch of the dataloader

        Returns:
            torch.tensor: result
        """
        x =  batch['x_num_past'].to(self.device)
        idx_target = batch['idx_target'][0]
        x_start = x[:,-1,idx_target].unsqueeze(1)
        #this is B,1,C
        
        #[B,L,C,1] remember the outoput size
        res = x_start.repeat(1,self.future_steps,1).unsqueeze(3)
        
        return res
    