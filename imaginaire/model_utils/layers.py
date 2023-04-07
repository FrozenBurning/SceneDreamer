import torch
import torch.nn as nn
import numpy as np


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)

class ConditionalHashGrid(nn.Module):
    def __init__(self, num_conv_blocks = 6):
        super(ConditionalHashGrid, self).__init__()
        self.sconv_head = nn.Conv2d(11, 8, kernel_size=3, stride=2, padding=1)
        self.hconv_head = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        conv_blocks = []
        cur_hdim = 16
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.fc1 = nn.Linear(cur_hdim, 16)
        self.fc2 = nn.Linear(16, 2)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, height_map, semantic_map):
        h = self.act(self.hconv_head(height_map))
        s = self.act(self.sconv_head(semantic_map))
        joint = torch.cat([h, s], dim=1)
        # interm = []
        # interm.append(joint.permute(0, 2, 3, 1).reshape(-1, 8))
        for layer in self.conv_blocks:
            out = self.act(layer(joint))
            # interm.append(out.permute(0, 2, 3, 1).reshape(-1, 8))
            joint = out
        
        out = out.permute(0, 2, 3, 1)
        out = torch.mean(out.reshape(out.shape[0], -1, out.shape[-1]), dim=1)
        cond = self.act(self.fc1(out))
        cond = torch.tanh(self.fc2(cond))
        return cond

class LightningMLP(nn.Module):
    r""" MLP with affine modulation."""

    def __init__(self, in_channels, style_dim, viewdir_dim, mask_dim=680,
                 out_channels_s=1, out_channels_c=3, hidden_channels=256,
                 use_seg=True):
        super(LightningMLP, self).__init__()

        self.use_seg = use_seg
        if self.use_seg:
            self.fc_m_a = nn.Linear(mask_dim, hidden_channels, bias=False)

        self.fc_viewdir = None
        if viewdir_dim > 0:
            self.fc_viewdir = nn.Linear(viewdir_dim, hidden_channels, bias=False)

        self.fc_1 = nn.Linear(in_channels, hidden_channels)

        self.fc_2 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_3 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_4 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)

        self.fc_sigma = nn.Linear(hidden_channels, out_channels_s)

        if viewdir_dim > 0:
            self.fc_5 = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.mod_5 = AffineMod(hidden_channels, style_dim, mod_bias=True)
        else:
            self.fc_5 = ModLinear(hidden_channels, hidden_channels, style_dim,
                                  bias=False, mod_bias=True, output_mode=True)
        self.fc_6 = ModLinear(hidden_channels, hidden_channels, style_dim, bias=False, mod_bias=True, output_mode=True)
        self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, raydir, z, m):
        r""" Forward network

        Args:
            x (N x H x W x M x in_channels tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            z (N x style_dim tensor): Style codes.
            m (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        b, h, w, n, _ = x.size()
        z = z[:, None, None, None, :]
        # print('style z', z.shape)
        # print('global enc:', global_enc.shape)
        f = self.fc_1(x)
        if self.use_seg:
            f = f + self.fc_m_a(m)
        # Common MLP
        f = self.act(f)
        f = self.act(self.fc_2(f, z))
        f = self.act(self.fc_3(f, z))
        f = self.act(self.fc_4(f, z))

        # Sigma MLP
        sigma = self.fc_sigma(f)

        # Color MLP
        if self.fc_viewdir is not None:
            f = self.fc_5(f)
            f = f + self.fc_viewdir(raydir)
            f = self.act(self.mod_5(f, z))
        else:
            f = self.act(self.fc_5(f, z))
        f = self.act(self.fc_6(f, z))
        c = self.fc_out_c(f)
        return sigma, c
    
class AffineMod(nn.Module):
    r"""Learning affine modulation of activation.

    Args:
        in_features (int): Number of input features.
        style_features (int): Number of style features.
        mod_bias (bool): Whether to modulate bias.
    """

    def __init__(self,
                 in_features,
                 style_features,
                 mod_bias=True
                 ):
        super().__init__()
        self.weight_alpha = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
        self.bias_alpha = nn.Parameter(torch.full([in_features], 1, dtype=torch.float))  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        if mod_bias:
            self.weight_beta = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
            self.bias_beta = nn.Parameter(torch.full([in_features], 0, dtype=torch.float))

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, 1, 1, , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        x = x * alpha

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            x = x + beta

        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x


class ModLinear(nn.Module):
    r"""Linear layer with affine modulation (Based on StyleGAN2 mod demod).
    Equivalent to affine modulation following linear, but faster when the same modulation parameters are shared across
    multiple inputs.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        style_features (int): Number of style features.
        bias (bool): Apply additive bias before the activation function?
        mod_bias (bool): Whether to modulate bias.
        output_mode (bool): If True, modulate output instead of input.
        weight_gain (float): Initialization gain
    """

    def __init__(self,
                 in_features,
                 out_features,
                 style_features,
                 bias=True,
                 mod_bias=True,
                 output_mode=False,
                 weight_gain=1,
                 bias_init=0
                 ):
        super().__init__()
        weight_gain = weight_gain / np.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) * weight_gain)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_alpha = nn.Parameter(torch.randn([in_features, style_features]) / np.sqrt(style_features))
        self.bias_alpha = nn.Parameter(torch.full([in_features], 1, dtype=torch.float))  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        self.output_mode = output_mode
        if mod_bias:
            if output_mode:
                mod_bias_dims = out_features
            else:
                mod_bias_dims = in_features
            self.weight_beta = nn.Parameter(torch.randn([mod_bias_dims, style_features]) / np.sqrt(style_features))
            self.bias_beta = nn.Parameter(torch.full([mod_bias_dims], 0, dtype=torch.float))

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, 1, 1, , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        w = self.weight.to(x.dtype)  # [O I]
        w = w.unsqueeze(0) * alpha  # [1 O I] * [B 1 I] = [B O I]

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            if not self.output_mode:
                x = x + beta

        b = self.bias
        if b is not None:
            b = b.to(x.dtype)[None, None, :]
        if self.mod_bias and self.output_mode:
            if b is None:
                b = beta
            else:
                b = b + beta

        # [B ? I] @ [B I O] = [B ? O]
        if b is not None:
            x = torch.baddbmm(b, x, w.transpose(1, 2))
        else:
            x = x.bmm(w.transpose(1, 2))
        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x
