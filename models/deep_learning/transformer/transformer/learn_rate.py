import math
import torch.optim as optim

class Scheduler():
    def __init__(self, optimizer: optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.step_num = 0

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self) -> float:
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class NoamScheduler(Scheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 d_model: int, 
                 warmup_steps: int = 4000) -> None:
        """
        Noam Learning Rate Scheduler
        
        Parameters:
            optimizer: The optimizer for which to schedule the learning rate
            d_model: Dimension of the model
            warmup_steps: Number of warmup steps
        """
        super().__init__(optimizer)
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_lr(self) -> float:
        return (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))


class CosineAnnealingWarmup(Scheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 max_lr: float, 
                 min_lr: float, 
                 warmup_steps: int, 
                 total_steps: int) -> None:
        """
        Cosine annealing with warmup learning rate scheduler
        
        Parameters:
            optimizer: The optimizer for which to schedule the learning rate
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.max_lr * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay


class LinearWarmupDecay(Scheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 max_lr: float, 
                 warmup_steps: int, 
                 total_steps: int) -> None:
        """
        Linear warmup followed by linear decay learning rate scheduler
        
        Parameters:
            optimizer: The optimizer for which to schedule the learning rate
            max_lr: Maximum learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.max_lr * self.step_num / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, decay_steps)
        return max(0.0, self.max_lr * (1 - progress))


class InverseSqrtDecay(Scheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 base_lr: float) -> None:
        """
        Inverse square root decay learning rate scheduler
        
        Parameters:
            optimizer: The optimizer for which to schedule the learning rate
            base_lr: Base learning rate
        """
        super().__init__(optimizer)
        self.base_lr = base_lr

    def get_lr(self) -> float:
        return self.base_lr / math.sqrt(self.step_num + 1)


class PolynomialDecay(Scheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 max_lr: float, 
                 end_lr: float, 
                 warmup_steps: int, 
                 total_steps: int, 
                 power: float = 1.0) -> None:
        """
        Polynomial decay learning rate scheduler with warmup
        
        Parameters:
            optimizer: The optimizer for which to schedule the learning rate
            max_lr: Maximum learning rate
            end_lr: Final learning rate after decay
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            power: Power of the polynomial decay
        """
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power

    def get_lr(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.max_lr * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.end_lr + (self.max_lr - self.end_lr) * ((1 - progress) ** self.power)


class ConstantWarmup(Scheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 max_lr: float, 
                 warmup_steps: int) -> None:
        """
        Constant learning rate with warmup scheduler
        
        Parameters:
            optimizer: The optimizer for which to schedule the learning rate
            max_lr: Maximum learning rate
            warmup_steps: Number of warmup steps
        """
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def get_lr(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.max_lr * self.step_num / self.warmup_steps
        return self.max_lr
