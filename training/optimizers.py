"""Custom optimizers + builder factory.

Domain layer. Depends on config (TrainConfig), not on training loop.
"""

import torch
import torch.optim as optim

from experiment.config import TrainConfig


# ── Lion ─────────────────────────────────────────────────────

class Lion(torch.optim.Optimizer):
    """Lion (EvoLved Sign Momentum) — PyTorch port from Google.

    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                 weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                if wd > 0:
                    p.mul_(1 - lr * wd)

                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


# ── Sophia ───────────────────────────────────────────────────

class Sophia(torch.optim.Optimizer):
    """Sophia — second-order clipped optimizer.

    Reference: https://arxiv.org/abs/2305.14342
    """

    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99),
                 weight_decay=0.01, rho=0.01):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, rho=rho)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            rho = group['rho']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg = state['exp_avg']
                hessian = state['hessian']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']

                grad.copy_(hessian).div_(bias2).sqrt_().clamp_(min=1e-15)
                grad.mul_(bias1)
                torch.div(exp_avg, grad, out=grad)
                grad.clamp_(-rho, rho)

                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(grad, alpha=-lr)
        return loss


# ── Builder ──────────────────────────────────────────────────

def build_optimizer(model, train_config: TrainConfig, device):
    """Build optimizer from model params + TrainConfig.

    Handles decay_linear_only param grouping (weight_decay on Linear only).
    """
    tc = train_config

    if tc.decay_linear_only:
        decay_params = []
        no_decay_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optim_groups = [
            {'params': decay_params, 'weight_decay': tc.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
    else:
        optim_groups = model.parameters()

    opt_name = tc.optimizer
    lr = tc.lr
    wd = tc.weight_decay

    if opt_name == 'nag':
        return optim.SGD(optim_groups, lr=lr, weight_decay=wd,
                         momentum=0.9, nesterov=True)
    if opt_name == 'sgd':
        return optim.SGD(optim_groups, lr=lr, weight_decay=wd, momentum=0.9)
    if opt_name == 'adamw_fused' and device.type == 'cuda':
        return optim.AdamW(optim_groups, lr=lr, weight_decay=wd, fused=True)
    if opt_name in ('adamw', 'adamw_fused'):
        return optim.AdamW(optim_groups, lr=lr, weight_decay=wd)
    if opt_name == 'lion':
        return Lion(optim_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.99))
    if opt_name == 'sophia':
        return Sophia(optim_groups, lr=lr, weight_decay=wd,
                      betas=(0.965, 0.99), rho=0.01)

    return optim.AdamW(optim_groups, lr=lr, weight_decay=wd)
