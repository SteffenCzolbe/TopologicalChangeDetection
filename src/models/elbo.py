import torch
import torch.nn as nn
import torch.nn.functional as nnf
import pytorch_lightning as pl
import torchreg
from src.semantic_loss import SemanticLossModel
import src.util as util
from typing import List


class ELBO(nn.Module):
    def __init__(self, data_dims, semantic_loss, init_recon_log_var, full_covar):
        super().__init__()
        self.ndims = torchreg.settings.get_ndims()
        self.data_dims = data_dims
        self.semantic_loss = semantic_loss
        self.full_covar = full_covar
        if self.semantic_loss:
            self.load_semantic_loss_model(model_path=semantic_loss)
        self.pi = torch.as_tensor(3.14159)

        if self.full_covar:
            # Cholesky decomposition lower triangular
            n = self.data_dims[0]
            init_covar_matrix = torch.diag(
                torch.ones(n)) * -init_recon_log_var  # initilize with diagonal precision matrix
            init_covar_matrix = init_covar_matrix.exp()
            weight = torch.cholesky(init_covar_matrix)
            self.L_full = torch.nn.parameter.Parameter(
                weight, requires_grad=True)

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
    def L(self):
        # return lower triangular
        return torch.tril(self.L_full)

    @property
    def covar(self):
        # return lower triangular
        if self.full_covar:
            L = self.L
            precision_m = torch.mm(L, L.T)
            return torch.inverse(precision_m)
        elif self.recon_log_var.shape == torch.Size([]):
            return self.recon_log_var.exp()
        else:
            return torch.diag(self.recon_log_var.exp())

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
        channel_cnt = sum(self.semantic_loss_model.net.unet.enc_feat)
        self.data_dims = (channel_cnt, *self.data_dims[1:])

    def loss(self, mu, log_var, transform, I0, I1, reduction='mean'):
        recon_loss = self.recon_loss(I0, I1, transform, reduction=reduction)
        kl_loss = self.kl_loss(
            mu, log_var, reduction=reduction)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

    def _transform_pyramid(self, pyramid, transform):
        """applies a transformation to a pyramid of features

        Args:
            pyramid ([type]): [description]
            transform ([type]): [description]
        """
        _, H, W, D = self.data_dims
        transformed_pyramid = []
        for features in pyramid:
            # scale fullsize transform down to smaller images
            scale_factor = features.shape[2] / H
            if self.ndims == 2:
                scaled_transform = nnf.interpolate(transform.squeeze(-1), scale_factor=scale_factor,
                                                   mode="bilinear", align_corners=True).unsqueeze(-1)
            else:
                scaled_transform = nnf.interpolate(transform, scale_factor=scale_factor,
                                                   mode="trilinear", align_corners=True)
            # scale transform vector length
            scaled_transform *= scale_factor
            transformed_pyramid.append(
                self.transformer(features, scaled_transform))
        return transformed_pyramid

    def _pyramid_to_fullsize(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Transforms a pyramid to a fullsize image.
        Scales values to keep channel-norm identical

        Args:
            pyramid ([type]): [description]
        """
        _, H, W, D = self.data_dims

        resized_pyramid = []
        for features in pyramid:
            # scale smaller pyramid features up to larger image size
            scale_factor = H / features.shape[2]
            volume_scale_factor = scale_factor**2 if self.ndims == 2 else scale_factor**3
            if self.ndims == 2:
                fullsize_features = nnf.interpolate(features.squeeze(-1), scale_factor=scale_factor,
                                                    mode="bilinear", align_corners=True).unsqueeze(-1)
            else:
                fullsize_features = nnf.interpolate(features, scale_factor=scale_factor,
                                                    mode="trilinear", align_corners=True)
            # re-scale normed values
            fullsize_features /= volume_scale_factor
            resized_pyramid.append(fullsize_features)

        # stack along channel dimension
        img = torch.cat(resized_pyramid, dim=1)
        return img

    def recon_loss(self, I0, I1, transform, reduction='mean'):
        # we implement the term pixel-whise, and mean over pixels if specified by the reduction
        # the scalar term is devided by factor n (canceled out), as it will be expanded (broadcasted) to size n during summation of the loss terms
        n = torch.prod(torch.tensor(self.data_dims[1:]))  # n pixels

        # Turn images into optional feature pyramids
        if self.semantic_loss:
            I0_pyramid = self.semantic_loss_model.extract_features(I0)
            I1_pyramid = self.semantic_loss_model.extract_features(I1)
        else:
            I0_pyramid = [I0]
            I1_pyramid = [I1]

        I01_pyramid = self._transform_pyramid(I0_pyramid, transform)

        if self.full_covar:
            loss = self.recon_loss_full_covar(I1_pyramid, I01_pyramid)
        else:
            loss = self.recon_loss_diag_covar(I1_pyramid, I01_pyramid)

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise Exception(f'reduction {reduction} not found.')

    def recon_loss_diag_covar(self, I1_pyramid, I01_pyramid):
        # recon loss with diagonal covariance matrix

        # memory-saving computation: do the loss calculation on each scale of the feature pyramid first, then upscale
        channel_from = 0
        loss_pyramid = []
        for I1, I01 in zip(I1_pyramid, I01_pyramid):
            diff = (I1 - I01)**2

            d = diff.shape[1]  # d channels on this pyramid level
            var = torch.exp(
                self.recon_log_var[channel_from:channel_from+d]).view(1, d, 1, 1, 1)
            log_var = self.recon_log_var[channel_from:channel_from+d]
            channel_from += d
            loss = (d/2 * torch.log(2 * self.pi)  # factor n multiplication via boradcasting during addition
                    + 0.5 * log_var.sum()  # factor n multiplication via boradcasting during addition
                    + 0.5 * torch.sum(diff / var, dim=1, keepdim=True))
            loss_pyramid.append(loss)

        # scale loss pyramid up to full size
        loss = self._pyramid_to_fullsize(loss_pyramid)

        # reduce loss to single channel
        return torch.sum(loss, dim=1, keepdim=True)

    def recon_loss_full_covar(self, I1_pyramid, I01_pyramid):
        # recon loss with full covar matrix,
        # covariance matrix via cholensky decomposition \Sigma^-1 = L L^T

        D = self.data_dims[0]  # D channels
        # transform pyramid to fullsize image (very memory intensive)
        I1 = self._pyramid_to_fullsize(I1_pyramid)
        I01 = self._pyramid_to_fullsize(I01_pyramid)

        # calculate ||v^T L||^2
        diff = I1 - I01
        diff = diff.permute(0, 2, 3, 4, 1).unsqueeze(
            4)  # reshape to BxHxWxDx1xC
        diff = torch.matmul(diff, self.L)
        diff = diff.squeeze(4).permute(
            0, 4, 1, 2, 3)  # reshape back to BxCxHxWxD
        # squared norm along channel dim
        diff = torch.sum(diff**2, dim=1, keepdim=True)

        loss = (D/2 * torch.log(2 * self.pi)  # factor n multiplication via boradcasting during addition
                # factor n multiplication via boradcasting during addition
                - 0.5 * torch.log(torch.diagonal(self.L) ** 2).sum()
                + 0.5 * diff)
        return loss

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
