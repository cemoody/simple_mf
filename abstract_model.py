import torch
import numpy as np
from random import shuffle
from torch import from_numpy
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import BatchSampler
from torch.utils.data import RandomSampler


class AbstractModel(pl.LightningModule):
    def step(self, batch, batch_nb, prefix='train', add_reg=True):
        input, target = batch
        prediction = self.forward(input)
        loss, log = self.loss(prediction, target)
        
        if add_reg:
            loss_reg, log_ = self.reg()
            loss = loss + loss_reg
            log.update(log_)
        log[f'{prefix}_loss'] = loss
        self.log(f"{prefix}_loss", loss)
        return {f'{prefix}_loss': loss, 'loss':loss, 'log': log}

    def training_step(self, batch, batch_nb):
        return self.step(batch, batch_nb, 'train')
    
    def test_step(self, batch, batch_nb):
        # Note that we do *not* include the regularization loss
        # at test time
        return self.step(batch, batch_nb, 'test', add_reg=False)    
    
    def validation_step(self, batch, batch_nb):
        return self.step(batch, batch_nb, 'val', add_reg=False)    
    
    def test_epoch_end(self, outputs):
        loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = {'test_loss': loss_mean}
        return {'avg_test_loss': loss_mean, 'log': log}

    def validation_epoch_end(self, outputs):
        loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': loss_mean}
        return {'avg_val_loss': loss_mean, 'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
