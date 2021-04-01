from torch.optim import lr_scheduler
import math

class LinearDecay(lr_scheduler._LRScheduler):
    """This class implements LinearDecay
    """

    def __init__(self, optimizer, num_epochs, start_epoch=0, min_lr=0, last_epoch=-1):
        """implements LinearDecay
        Parameters:
        ----------
        """
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return self.base_lrs
        return [base_lr - ((base_lr - self.min_lr) / self.num_epochs) * (self.last_epoch - self.start_epoch) for
                base_lr in self.base_lrs]