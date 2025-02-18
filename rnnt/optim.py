from torch import optim


class Optimizer(object):
    def __init__(self, model, config):
        self.config = config
        self.optimizer = build_optimizer(model, config)
        self.global_step = 1
        self.current_epoch = 0
        self.lr = config.lr
        self.decay_ratio = config.decay_ratio
        self.epoch_decay_flag = False

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def epoch(self):
        self.current_epoch += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def decay_lr(self):
        self.lr = max(self.decay_ratio * self.lr, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr


def get_optim_groups(model, weight_decay):
    parameters = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in parameters if p.dim() >= 2]
    nondecay_params = [p for p in parameters if p.dim() < 2]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nondecay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model, config):
    params = get_optim_groups(model, config.weight_decay)
    if config.type == "adamw":
        return optim.AdamW(
            params=params,
            weight_decay=config.weight_decay,
            lr=config.lr,
            betas=tuple(map(float, config.betas.strip('()').split(','))),
            eps=float(config.eps),
            fused=config.fused,
        )
    elif config.type == "sgd":
        return optim.SGD(
            params=params,
            weight_decay=config.weight_decay,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    else:
        raise NotImplementedError
