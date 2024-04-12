class TrainingConfig:

    def __init__(self, lr: float, epochs: int, batch_size: int, logits_loss: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.logits_loss = logits_loss

    def set_lr(self, lr: float):
        self.lr = lr

    def set_epochs(self, epochs: int):
        self.epochs = epochs

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_logits_loss(self, logits_loss: bool):
        self.logits_loss = logits_loss

    def get_lr(self) -> float:
        return self.lr

    def get_epochs(self) -> int:
        return self.epochs

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_logits_loss(self) -> bool:
        return self.logits_loss
