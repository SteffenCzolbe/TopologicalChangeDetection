import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchreg
from src.semantic_loss import SemanticLossModel
import src.util as util


class ELBO(nn.Module):
    def __init__(self, data_dims, semantic_loss, init_recon_log_var):
        super().__init__()
        self.ndims = torchreg.settings.get_ndims()
        self.data_dims = data_dims
        self.semantic_loss = semantic_loss
        if self.semantic_loss:
            self.load_semantic_loss_model(model_path=semantic_loss)
        self.pi = torch.as_tensor(3.14159)

        if self.data_dims[0] > 1:
            init_recon_log_var = [init_recon_log_var] * self.data_dims[0]
        self.recon_log_var = torch.nn.parameter.Parameter(
            torch.as_tensor(init_recon_log_var, dtype=torch.float32), requires_grad=True)
        self.running_mean_log_alpha = nn.BatchNorm1d(
            1, track_running_stats=True)
        self.running_mean_log_beta = nn.BatchNorm1d(
            1, track_running_stats=True)
        self.grad_norm = torchreg.metrics.GradNorm(
            penalty="l2", reduction="none")
        self.transformer = torchreg.nn.SpatialTransformer()

    @property
    def log_alpha(self):
        return self.running_mean_log_alpha.running_mean.squeeze()

    @property
    def log_beta(self):
        return self.running_mean_log_beta.running_mean.squeeze()

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
        # the scalar term is devided by factor n (canceled out), as it will be expanded (broadcasted) to size n during summation of the loss terms
        n = torch.prod(torch.tensor(self.data_dims[1:]))  # n pixels
        D = self.data_dims[0]  # D channels

        if self.semantic_loss:
            I0 = self.semantic_loss_model.augment_image(I0)
            I1 = self.semantic_loss_model.augment_image(I1)

        I01 = self.transformer(I0, transform)

        diff = (I1 - I01)**2

        var = torch.exp(self.recon_log_var).view(1, D, 1, 1, 1)
        log_var = self.recon_log_var

        loss = (D/2 * torch.log(2 * self.pi)  # factor n multiplication via boradcasting during addition
                + 0.5 * log_var.sum()  # factor n multiplication via boradcasting during addition
                + 0.5 * torch.sum(diff / var, dim=1, keepdim=True))

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
        n = torch.prod(torch.tensor(self.data_dims[1:]))  # n pixels
        D = self.ndims  # D dimensions
        neighborhood_size = 2 * self.ndims  # degree matrix

        # log_det_p and translation terms are devided by factor n,
        # as the scalar will be expanded (broadcasted) to size n during summation of the loss terms
        # we sum aross the transformation channels for all terms
        var = torch.sum(torch.exp(log_var), dim=1, keepdim=True)
        diffusion_reg = self.grad_norm(
            mu) + self.grad_norm(mu.flip(dims=[2, 3, 4])).flip(dims=[2, 3, 4])
        translation_component = 1 / n * \
            torch.sum(torch.sum(mu, dim=[2, 3, 4],
                                keepdim=True)**2, dim=1, keepdim=True)
        log_det_q = torch.sum(log_var, dim=1, keepdim=True)

        if self.training:
            # use expectations over batch
            alpha = (n-1) * D / expect(neighborhood_size * var + diffusion_reg)
            beta = n**2 * D / expect(var + translation_component)
            log_alpha = torch.log(alpha)
            log_beta = torch.log(beta)
            # add to running mean
            self.running_mean_log_alpha(log_alpha.expand(1, 1, 2))
            self.running_mean_log_beta(log_beta.expand(1, 1, 2))
        else:
            # take alpha, beta from running means
            alpha = torch.exp(self.log_alpha)
            beta = torch.exp(self.log_beta)
            log_alpha = self.log_alpha
            log_beta = self.log_beta

        log_det_p = 1/n * (
            - (n-1) * D * log_alpha
            - D * log_beta
        )

        if self.training:
            # calculate the shortened loss term, where some expectations cancel each other out
            loss = 0.5 * (
                log_det_p
                - log_det_q
            )
        else:
            # calculate the whole loss term using alpha, beta
            loss = 0.5 * \
                (log_det_p
                 - log_det_q
                 + (alpha * neighborhood_size + beta / (n**2)) * var
                 + alpha * diffusion_reg
                 + beta / (n**2) * translation_component
                 )

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')
