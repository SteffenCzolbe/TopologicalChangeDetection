import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchreg
from src.semantic_loss import SemanticLossModel
import src.util as util


class ELBO(nn.Module):
    def __init__(self, data_dims, use_analytical_prior, use_analytical_recon, semantic_loss, init_prior_log_alpha, init_prior_log_beta, trainable_prior, init_recon_log_var, trainable_recon_var):
        super().__init__()
        self.ndims = torchreg.settings.get_ndims()
        self.data_dims = data_dims
        self.use_analytical_prior = use_analytical_prior
        self.use_analytical_recon = use_analytical_recon
        self.semantic_loss = semantic_loss
        if self.semantic_loss:
            self.load_semantic_loss_model(model_path=semantic_loss)
        self.pi = torch.as_tensor(3.14159)

        self.prior_log_alpha = torch.nn.parameter.Parameter(
            torch.as_tensor(init_prior_log_alpha, dtype=torch.float32), requires_grad=trainable_prior)
        self.prior_log_beta = torch.nn.parameter.Parameter(
            torch.as_tensor(init_prior_log_beta, dtype=torch.float32), requires_grad=trainable_prior)
        if self.data_dims[0] > 1:
            init_recon_log_var = [init_recon_log_var] * self.data_dims[0]
        self.recon_log_var = torch.nn.parameter.Parameter(
            torch.as_tensor(init_recon_log_var, dtype=torch.float32), requires_grad=trainable_recon_var)
        self.grad_norm = torchreg.metrics.GradNorm(
            penalty="l2", reduction="none")
        self.transformer = torchreg.nn.SpatialTransformer()

    def load_semantic_loss_model(self, model_path):
        # load semantic loss model
        model_checkpoint = util.get_checkoint_path_from_logdir(model_path)
        self.semantic_loss_model = SemanticLossModel.load_from_checkpoint(
            model_checkpoint)
        util.freeze_model(self.semantic_loss_model)
        # adjust image channels for augmented data
        channel_cnt = sum(self.semantic_loss_model.net.enc_feat)
        self.data_dims = (channel_cnt, *self.data_dims[1:])

    def loss(self, mu, log_var, transform, I0, I1, reduction='mean'):
        recon_loss = self.recon_loss(I0, I1, transform, reduction=reduction)
        kl_loss = self.kl_loss(
            mu, log_var, reduction=reduction)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

    def recon_loss(self, I0, I1, transform, reduction='mean'):
        # we implement the term pixel-whise, and mean over pixels if specified by the reduction
        # the scalar term is devided by factor p (canceled out), as it will be expanded (broadcasted) to size p during summation of the loss terms
        p = torch.prod(torch.tensor(self.data_dims[1:]))  # p pixels
        c = self.data_dims[0]  # c channels

        def expect(t):
            # sum spatial dimensions
            t = t.sum(dim=[2, 3, 4], keepdim=True)
            # mean (expectation) over the batch and channels
            return t.mean(dim=[0, 1], keepdim=True)

        if self.semantic_loss:
            I0 = self.semantic_loss_model.augment_image(I0)
            I1 = self.semantic_loss_model.augment_image(I1)

        I01 = self.transformer(I0, transform)

        diff = (I1 - I01)**2

        if self.use_analytical_recon:
            var = 1 / (p * expect(diff))
            log_var = torch.log(var)
        else:
            var = torch.exp(self.recon_log_var).view(
                1, self.data_dims[0], 1, 1, 1)
            log_var = self.recon_log_var

        loss = 0.5 * (c * torch.log(2 * self.pi) + log_var.sum()) \
            + 0.5 * torch.sum(1/var * diff, dim=1, keepdim=True)

        if self.use_analytical_recon:
            # set parameter for logging
            self.recon_log_var = torch.nn.Parameter(
                log_var.squeeze(), requires_grad=False)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')

    def kl_loss(self, mu, log_var, reduction='mean'):
        def expect(t):
            # sum spatial dimensions
            t = t.sum(dim=[1, 2, 3, 4])
            # mean (expectation) over the batch
            return t.mean()

        # we implement the term pixel-whise, summing over the channel dimension and mean over pixels if specified by the reduction
        p = torch.prod(torch.tensor(self.data_dims[1:]))  # p pixels
        n = self.ndims * p  # n displacement-vector components
        degree = 2 * self.ndims  # degree matrix

        # log_det_p and translation terms are devided by factor p,
        # as the scalar will be expanded (broadcasted) to size p during summation of the loss terms
        # we mean aross the transformation channels for all terms
        var = torch.sum(torch.exp(log_var), dim=1, keepdim=True)
        diffusion_reg = self.grad_norm(
            mu) + self.grad_norm(mu.flip(dims=[2, 3, 4])).flip(dims=[2, 3, 4])
        translation_component = 1 / p * \
            torch.sum(mu, dim=[1, 2, 3, 4], keepdim=True)**2 / self.ndims
        log_det_q = torch.sum(log_var, dim=1, keepdim=True)

        if self.use_analytical_prior:
            alpha = (n-1) / expect(degree * var + diffusion_reg)
            beta = n**2 / expect(var + translation_component)
            log_alpha = torch.log(alpha)
            log_beta = torch.log(beta)
        else:
            alpha = torch.exp(self.prior_log_alpha)
            beta = torch.exp(self.prior_log_beta)
            log_alpha = self.prior_log_alpha
            log_beta = self.prior_log_beta

        log_det_p = 1/p * (
            - (n-1) * torch.log(alpha)
            - torch.log(beta)
        )

        if self.use_analytical_prior:
            # analytical solution for alpha, beta
            loss = 0.5 * self.ndims * \
                (log_det_p
                 - log_det_q
                 + (n-1) * (degree * var + diffusion_reg)
                 / expect(degree * var + diffusion_reg)
                 + (var + translation_component)
                 / expect(var + translation_component)
                 )
            # set alpha, beta for logging
            self.prior_log_alpha = torch.nn.Parameter(
                torch.log(torch.as_tensor(alpha)), requires_grad=False)
            self.prior_log_beta = torch.nn.Parameter(
                torch.log(torch.as_tensor(beta)), requires_grad=False)
        else:
            # parameterized by alpha, beta
            loss = 0.5 * self.ndims * \
                (log_det_p
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
