import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchreg


class ELBO(nn.Module):
    def __init__(self, data_dims, init_prior_log_alpha, init_prior_log_beta, init_recon_log_var, param_size=(1,)):
        super().__init__()
        self.ndims = torchreg.settings.get_ndims()
        self.data_dims
        self.prior_log_alpha = torch.nn.parameter.Parameter(
            init_prior_log_alpha * torch.ones(param_size), requires_grad=True)
        self.prior_log_beta = torch.nn.parameter.Parameter(
            init_prior_log_beta * torch.ones(param_size), requires_grad=True)
        self.recon_log_var = torch.nn.parameter.Parameter(
            init_recon_log_var * torch.ones(param_size), requires_grad=True)
        self.grad_norm = torchreg.metrics.GradNorm(
            penalty="l2", reduction="none")

    def loss(self, mu, log_var, I01, I1, reduction='mean'):
        recon_loss = self.recon_loss(mu, I01, I1, reduction=reduction)
        kl_loss = self.kl_loss(mu, log_var, reduction=reduction)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

    def recon_loss(self, mu, I01, I1, reduction='mean'):
        var = torch.exp(self.recon_log_var)
        loss = 0.5 * torch.log(2 * 3.14159 * var) + 1/(2 * var) * \
            torch.mean((I1 - I01)**2, dim=1, keepdim=True)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')

    def kl_loss(self, mu, log_var, reduction='mean'):
        # we implement the term pixel-whise
        n = self.ndims * torch.prod(self.data_dims[1:])
        degree = 2*self.ndims
        alpha = torch.exp(self.prior_log_alpha)
        beta = torch.exp(self.prior_log_beta)

        a = - (n - 1) * self.prior_log_alpha / n - self.prior_log_beta / n
        b = - torch.sum(log_var, dim=1, keepdim=True)
        c = torch.sum(torch.exp(log_var), dim=1, keepdim=True) * \
            (alpha * degree + beta / (n**2))
        d = alpha * (self.grad_norm(mu) +
                     self.grad_norm(mu.flip(dims=[2, 3, 4])))
        e = beta * torch.mean(mu, dim=1, keepdim=True)**2

        loss = a+b+c+d+e

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')
