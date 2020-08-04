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
    def save_data(self, train_x, train_y, test_x, test_y, train_d=None, test_d=None):
        if train_d is None:
            self.train_arrs = [from_numpy(x) for x in [train_x, train_y]]
            self.test_arrs = [from_numpy(x) for x in [test_x, test_y]]
        else:
            self.train_arrs = [from_numpy(x) for x in [train_x, train_y]] + [train_d]
            self.test_arrs = [from_numpy(x) for x in [test_x, test_y]] +[test_d]

    def step(self, batch, batch_nb, prefix='train', add_reg=True):
        input, target = batch
        prediction = self.forward(input)
        loss, log = self.likelihood(prediction, target)
        
        if add_reg:
            loss_reg, log_ = self.reg()
            loss = loss + loss_reg
            log.update(log_)
        log[f'{prefix}_loss'] = loss
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
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = {'val_loss': test_loss_mean}
        return {'avg_test_loss': test_loss_mean, 'log': log}

    def validation_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': test_loss_mean}
        return {'avg_val_loss': test_loss_mean, 'log': log}

    def dataloader(self, is_train=True):
        if is_train:
            dataset = TensorDataset(*self.train_arrs)
        else:
            dataset = TensorDataset(*self.test_arrs)
        bs = BatchSampler(RandomSampler(dataset), 
                          batch_size=self.batch_size, drop_last=False)
        return DataLoader(dataset, batch_sampler=bs, num_workers=8)
    
    def train_dataloader(self):
        return self.dataloader(is_train=True)

    def test_dataloader(self):
        return self.dataloader(is_train=False)

    def val_dataloader(self):
        return self.dataloader(is_train=False)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)