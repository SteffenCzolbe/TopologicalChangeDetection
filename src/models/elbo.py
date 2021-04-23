import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchreg


class ELBO(nn.Module):
    def __init__(self, data_dims, use_analytical_solution, init_prior_log_alpha, trainable_alpha, init_prior_log_beta, trainable_beta, init_recon_log_var, trainable_recon_var):
        """The evidence lower bound

        Args:
            data_dims (tuple): dimension of the data: C, H, W, D
            use_analytical_solution (bool): True: use analytical solution for alpha, beta. False: use parameters alpha, beta.
            init_prior_log_alpha (float): initialization.
            trainable_alpha (bool): should the parameter be trained?
            init_prior_log_beta (float): initialization.
            trainable_alpha (bool): should the parameter be trained?
            init_recon_log_var (float): initialization.
            trainable_alpha (bool): should the parameter be trained?
        """
        super().__init__()
        self.ndims = torchreg.settings.get_ndims()
        self.data_dims = data_dims
        self.use_analytical_solution = use_analytical_solution
        self.prior_log_alpha = torch.nn.parameter.Parameter(
            torch.as_tensor(init_prior_log_alpha, dtype=torch.float32), requires_grad=trainable_alpha)
        self.prior_log_beta = torch.nn.parameter.Parameter(
            torch.as_tensor(init_prior_log_beta, dtype=torch.float32), requires_grad=trainable_beta)
        self.recon_log_var = torch.nn.parameter.Parameter(
            torch.as_tensor(init_recon_log_var, dtype=torch.float32), requires_grad=trainable_recon_var)
        self.pi = torch.as_tensor(3.14159)
        self.grad_norm = torchreg.metrics.GradNorm(
            penalty="l2", reduction="none")

    def loss(self, mu, log_var, I01, I1, reduction='mean'):
        recon_loss = self.recon_loss(I01, I1, reduction=reduction)
        kl_loss = self.kl_loss(
            mu, log_var, use_analytical_solution=self.use_analytical_solution, reduction=reduction)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

    def recon_loss(self, I01, I1, reduction='mean'):
        # we implement the term pixel-whise, and mean over pixels if specified by the reduction
        # the scalar term is devided by factor p (canceled out), as it will be expanded (broadcasted) to size p during summation of the loss terms
        var = torch.exp(self.recon_log_var)
        loss = 0.5 * (torch.log(2 * self.pi) + self.recon_log_var) \
            + 1/(2 * var) * torch.mean((I1 - I01)**2, dim=1, keepdim=True)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')

    def kl_loss(self, mu, log_var, use_analytical_solution, reduction='mean'):
        def expect(t):
            # take the expectation over the minibatch
            B = t.shape[0]
            return t.mean(dim=0, keepdim=True).expand(B, -1, -1, -1, -1)

        # we implement the term pixel-whise, summing over the channel dimension and mean over pixels if specified by the reduction
        p = torch.prod(torch.tensor(self.data_dims[1:]))  # p pixels
        n = self.ndims * p  # n displacement-vector components
        degree = 2 * self.ndims  # degree matrix
        alpha = torch.exp(self.prior_log_alpha)
        beta = torch.exp(self.prior_log_beta)

        # log_det_p_parameterized and translation terms are devided by factor p,
        # as the scalar will be expanded (broadcasted) to size p during summation of the loss terms
        var = torch.sum(torch.exp(log_var), dim=1, keepdim=True)
        diffusion_reg = self.grad_norm(
            mu) + self.grad_norm(mu.flip(dims=[2, 3, 4]))
        translation_component = 1 / p * torch.sum(mu, dim=[1, 2, 3, 4])**2
        log_det_p_parameterized = (
            - (n - 1) / p * self.prior_log_alpha
            - self.prior_log_beta / p
        )
        log_det_p_analytical = (
            (n-1) * torch.log(expect(degree * var + diffusion_reg))
            + torch.log(expect(var + translation_component))
        )
        log_det_q = torch.sum(log_var, dim=1, keepdim=True)

        if use_analytical_solution:
            # analytical solution for alpha, beta
            loss = 0.5 * (log_det_p_analytical
                          - log_det_q
                          + (n-1) * (degree * var + diffusion_reg)
                          / expect(degree * var + diffusion_reg)
                          + (var + translation_component)
                          / expect(var + translation_component)
                          )
        else:
            # parameterized by alpha, beta
            loss = 0.5 * (log_det_p_parameterized
                          - log_det_q
                          + (alpha * degree + beta / (n**2)) * var
                          + alpha * diffusion_reg
                          + beta / (n**2) * translation_component
                          )

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')
