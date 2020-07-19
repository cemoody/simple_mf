import torch
import numpy as np
from random import shuffle
import pytorch_lightning as pl


def chunks(*arrays, size):
    """Yield successive n-sized chunks from l."""
    starts = list(range(0, len(arrays[0]), size))
    # always visit your data in a random order!
    shuffle(starts)
    for i in starts:
        arrs = [arr[i: i + size] for arr in arrays]
        # convert numpy arrays to torch arrays
        arrs_torch = []
        for arr in arrs:
            if arr.dtype != np.dtype('O'):
                tarr = torch.from_numpy(arr)
                arrs_torch.append(tarr)
            else:
                arrs_torch.append(arr)
        yield arrs_torch[:-1], arrs_torch[-1]


class AbstractModel(pl.LightningModule):
    def save_data(self, train_x, train_y, test_x, test_y, train_d=None, test_d=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.train_d = train_d
        self.test_d = test_d

    def training_step(self, batch, batch_nb):
        input, target = batch
        prediction = self.forward(input)
        loss = self.likelihood(prediction, target) + self.prior()
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input, target = batch
        prediction = self.forward(input)
        # Note that we do *not* include the regularization / prior loss
        # at test time
        loss = self.likelihood(prediction, target)
        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}
    
    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}
    
    def train_dataloader(self):
        if self.train_d is None:
            return chunks(self.train_x, self.train_y, size=self.batch_size)
        else:
            return chunks(self.train_x, self.train_d, self.train_y, size=self.batch_size)
        
    def test_dataloader(self):
        if self.test_d is None:
            return chunks(self.test_x, self.test_y, size=self.batch_size)
        else:
            return chunks(self.test_x, self.test_d, self.test_y, size=self.batch_size)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
