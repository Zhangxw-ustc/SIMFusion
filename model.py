import os
from pickle import Unpickler
import numpy as np
import cv2
from audioop import bias
from locale import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from utils.CDC import cdcconv
import time
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
CE = torch.nn.BCELoss(reduction='sum')
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



#----------------------------------------------
class FRNet(nn.Module):
    """
    Feature refinement network：
    (1) IEU
    (2) CSGate
    """
    def __init__(self, in_features, num_heads, weight_type="bit", num_layers=1, att_size=8, mlp_layer=256):
        """
        :param field_length: field_length
        :param embed_dim: embedding dimension
        type: bit or vector
        """
        super(FRNet, self).__init__()
        # IEU_G computes complementary features.
        # self.IEU_G = Attention(embed_dim, num_heads=att_size, b)

        # self.IEU_W = Attention(embed_dim, num_heads=att_size, b)
        
        self.IEU_G = IEU(in_features, num_heads, weight_type="bit")

        # IEU_W computes bit-level or vector-level weights.
        self.IEU_W = IEU(in_features, num_heads, weight_type=weight_type)
        self.conv = nn.Conv2d(in_features, 1, 1)

    def forward(self, x_dec, x_seg):   # b c h w
        com_feature = self.IEU_G(x_dec)
        wegiht_matrix = torch.sigmoid(self.IEU_W(x_seg))
        # CSGate
        # x_out = com_feature * (torch.tensor(1.0) - wegiht_matrix) + x_dec

        x_out = x_seg * wegiht_matrix + com_feature * (torch.tensor(1.0) - wegiht_matrix) + x_dec
        x_out = self.conv(x_out)
        return x_out, x_dec, x_seg, com_feature, wegiht_matrix


class IEU(nn.Module):
    """
    Information extraction Unit (IEU) for FRNet
    (1) Self-attention
    (2) DNN
    """
    def __init__(self, in_features, num_heads, weight_type="bit"):
        """
        :param field_length:
        :param embed_dim:
        :param type: vector or bit
        :param bit_layers:
        :param att_size:
        :param mlp_layer:
        """
        super(IEU,self).__init__()
        self.in_features = in_features
        
        self.weight_type = weight_type
        self.num_heads = num_heads

        # Self-attention unit, which is used to capture cross-feature relationships.
        self.vector_info = Attention(dim=self.in_features, num_heads=self.num_heads, bias=False)

        #  contextual information extractor(CIE), we adopt MLP to encode contextual information.
        # mlp_layers = [mlp_layer for _ in range(bit_layers)]
        self.mlps = Mlp(self.in_features, hidden_features=None, ffn_expansion_factor = 2, bias = False)
        # self.bit_projection = nn.Linear(mlp_layer, embed_dim)
        self.activation = nn.ReLU()
        # self.activation = nn.PReLU()


    def forward(self,x):
        """
        :param x_emb: B,F,E
        :return: B,F,E (bit-level weights or complementary fetures)
                 or B,F,1 (vector-level weights)
        """

        # （1）self-attetnion unit
        x_vector = self.vector_info(x)  # b, c, h, w

        # (2) CIE unit
        x_bit = self.mlps(x)  #bhw, c
        # x_bit = self.bit_projection(x_bit).unsqueeze(-1).unsqueeze(-1) # B, C, 1, 1
        x_bit = self.activation(x_bit)

        # （3）integration unit
        x_out = x_bit * x_vector

        if self.weight_type == "vector":
            # To compute vector-level importance in IEU_W
            x_out = torch.sum(x_out,dim=2,keepdim=True).unsqueeze(-1)
            # B,F,1
            return x_out

        return x_out


class SelfAttentionIEU(nn.Module):
    def __init__(self, embed_dim, att_size=20):
        """
        :param embed_dim:
        :param att_size:
        """
        super(SelfAttentionIEU, self).__init__()
        self.embed_dim = embed_dim
        self.trans_Q = nn.Linear(embed_dim,att_size)
        self.trans_K = nn.Linear(embed_dim,att_size)
        self.trans_V = nn.Linear(embed_dim,att_size)
        self.projection = nn.Linear(att_size,embed_dim)
        # self.scale = 1.0/ torch.LongTensor(embed_dim)
        # self.scale = torch.sqrt(1.0 / torch.tensor(embed_dim).float())
        # self.dropout = nn.Dropout(0.5)
        # self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self,x, scale=None):
        """
        :param x: B,F,E
        :return: B,F,E
        """
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # B,F,F
        attention_score = F.softmax(attention, dim=-1)
        context = torch.matmul(attention_score, V)
        # Projection
        context = self.projection(context)
        # context = self.layer_norm(context)
        return context


class MultiLayerPerceptronPrelu(torch.nn.Module):
    def __init__(self, input_dim, output_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for output_dim in output_dims:
            layers.append(torch.nn.Linear(input_dim, output_dim))
            layers.append(torch.nn.BatchNorm2d(output_dim))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout2d(p=dropout))
            input_dim = output_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        return self.mlp(x)
#----------------------------------------------

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=32, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################



class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, cat = True):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.transconv_same = nn.ConvTranspose2d(in_channels, in_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cat = cat

    def forward(self, x, y):
        
        if(self.cat):
            x = self.transconv(x)
            x = torch.cat((x, y), dim=1)
        else:
            x = self.transconv_same(x)
    
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class DecomNet(nn.Module):
    def __init__(self,
                 in_chan = 1,
                 out_chan = 1,
                 dim=256,
                 num_blocks=[1, 1],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',):
        super(DecomNet, self).__init__()
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.patch_embed = OverlapPatchEmbed(in_chan, dim)
        self.down1 = Downsample_block(in_chan, 32)
        self.down2 = Downsample_block(32, 64)
        self.down3 = Downsample_block(64, 128)
        # self.down4 = Downsample_block(128, 256)

        # self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        # self.up4 = Upsample_block(512, 256)
        self.up3 = Upsample_block(256, 128)
        self.up2 = Upsample_block(128, 64)
        self.up1 = Upsample_block(64, 32)

        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1 , 1)

        self.conv5 = nn.Conv2d(32, 32, 3, padding = 1)
        # self.up4_nocat = Upsample_block(512, 256, cat = False)
        self.up3_nocat = Upsample_block(256, 128, cat = False)
        self.up2_nocat = Upsample_block(128, 64, cat = False)
        self.up1_nocat = Upsample_block(64, 32, cat = False)

        # self.outconv = nn.Conv2d(32, out_chan, 1)
        # self.outconv_nocat = nn.Conv2d(32, 3, padding = 1)
        self.outconv_third = nn.Conv2d(64,1,1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        

        # self.conv_last = nn.Conv2d(3,1,padding = 1)

    def forward(self, x_orig):
        x, y1 = self.down1(x_orig)    # 32
        x, y2 = self.down2(x)         # 64
        x, y3 = self.down3(x)
        # x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))    # 128
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))    # 128
        x = self.encoder_level1(x)
# --------------------------------------------------------------------------
        # x1 = self.up4(x, y4)
        x1 = self.up3(x, y3)
        x1 = self.up2(x1, y2)    # 64
        x1_32 = self.up1(x1, y1)    # 32
        x1_16 = self.conv3(x1_32)    #16
        x1 = self.conv4(x1_16)      #16 ---- > 1
        x1 = self.sigmoid(x1) 

        # x2 = self.up4(x, y4)
        # x2 = self.up3(x2, y3)
        # x2 = self.up2(x2, y2)
        # x2_tmp = self.up1(x2, y1)
        # x2 = self.outconv(x2_tmp)
        # x2 = self.sigmoid(x2)

        x2 =self.conv5(y1)   #32
        x2 = self.outconv_third(torch.cat((x2, x1_32), dim=1))   # 64 --> 1
        # x2 = self.outconv_third(x2)
        x_out = self.softplus(x2)

        return x1, x_out, x1_16
# --------------------------------------------------------------------------

class Prediction(nn.Module):
    def __init__(self, num_channels):
        super(Prediction, self).__init__()
        self.num_layers = 4
        self.in_channel = num_channels
        self.kernel_size = 9
        self.num_filters = 64

        self.layer_in = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                  kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in.weight.data)
        self.lam_in = nn.Parameter(torch.Tensor([0.01]))

        self.lam_i = []
        self.layer_down = []
        self.layer_up = []
        for i in range(self.num_layers):
            down_conv = 'down_conv_{}'.format(i)
            up_conv = 'up_conv_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2.weight.data)
            setattr(self, down_conv, layer_2)
            self.layer_down.append(getattr(self, down_conv))
            layer_3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_3.weight.data)
            setattr(self, up_conv, layer_3)
            self.layer_up.append(getattr(self, up_conv))

            lam_ = nn.Parameter(torch.Tensor([0.01]))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod):
        p1 = self.layer_in(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam_in))

        for i in range(self.num_layers):
            p3 = self.layer_down[i](tensor)
            p4 = self.layer_up[i](p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam_i[i]))
        return tensor

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.channel = 32
        self.kernel_size = 9
        self.filters = 64
        self.conv_1 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        self.conv_1_1 = nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_1.weight.data)
        self.conv_2 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        self.conv_2_1 = nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_2.weight.data)
        self.conv_3 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        self.conv_3_1 = nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_3.weight.data)

    def forward(self, u,v,z):
        rec_u_32 = self.conv_1(u) # 32
        rec_u_1 = self.conv_1_1(rec_u_32)

        rec_z_32 = self.conv_2(z)
        rec_z_1 = self.conv_2_1(rec_z_32)

        rec_v_32 = self.conv_3(v)
        rec_v_1 = self.conv_3_1(rec_v_32)

        z_rec = rec_u_1 + rec_z_1 + rec_v_1
        return z_rec, rec_u_1, rec_z_1, rec_v_1, rec_z_32


class CUNet(nn.Module):
    def __init__(self):
        super(CUNet, self).__init__()
        self.channel = 1
        self.num_filters = 64
        self.kernel_size = 9
        self.net_u = Prediction(num_channels=self.channel)
        self.conv_u = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_u.weight.data)
        self.net_v = Prediction(num_channels=self.channel)
        self.conv_v = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_v.weight.data)
        self.net_z = Prediction(num_channels=2 * self.channel)
        self.decoder = decoder()

    def forward(self, x, y):
        u = self.net_u(x)
        v = self.net_v(y)

        p_x = x - self.conv_u(u)
        p_y = y - self.conv_v(v)
        p_xy = torch.cat((p_x, p_y), dim=1)

        z = self.net_z(p_xy)
        z_rec, rec_u_1, rec_z_1, rec_v_1, rec_z_32 = self.decoder(u,v,z)
        return z_rec, rec_u_1, rec_z_1, rec_v_1, rec_z_32


class SegNet(nn.Module):
    def __init__(self,
                 in_chan = 2,
                 out_chan = 1):
        super(SegNet, self).__init__()
        self.down1 = Downsample_block(in_chan, 32)
        self.down2 = Downsample_block(32, 64)
        self.down3 = Downsample_block(64, 128)
        # self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(256, 128)
        self.up2 = Upsample_block(128, 64)
        self.up1 = Upsample_block(64, 32)
        self.outconv = nn.Conv2d(32, out_chan, 1)
        self.outconvp1 = nn.Conv2d(32, out_chan, 1)
        self.outconvm1 = nn.Conv2d(32, out_chan, 1)
        self.fusion = FRNet(32, 8)

    def forward(self, x, x_decom1, x_decom2):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        # x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        # x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)   # 32
        x1 = self.outconv(x)  # 32 ----> 1
        out = self.fusion(torch.cat([x_decom1, x_decom2], dim=1), x)
        return x1, out

class Segmentation(nn.Module):
    def __init__(self,
                 in_chan = 2,
                 out_chan = 1):
        super(Segmentation, self).__init__()
        self.down1 = Downsample_block(in_chan, 32)
        self.down2 = Downsample_block(32, 64)
        self.down3 = Downsample_block(64, 128)
        self.down4 = Downsample_block(128, 256)
        # self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.up4 = Upsample_block(512, 256)
        
        self.up3 = Upsample_block(256, 128)
        self.up2 = Upsample_block(128, 64)
        self.up1 = Upsample_block(64, 32)
        self.outconv1 = nn.Conv2d(32, 16, 1)
        self.outconv2 = nn.Conv2d(16, out_chan, 1)

        # self.outconvm1 = nn.Conv2d(32, out_chan, 1)
        # self.fusion = FRNet(32, 8)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x_32 = self.up1(x, y1)   
        x_16 = self.outconv1(x_32)  # 32 ----> 16
        x_1 = self.outconv2(x_16)  # 16 ----> 1
        # out = self.fusion(torch.cat([x_decom1, x_decom2], dim=1), x)
        return x_1, x_16, x_32

class Fusion(nn.Module):
    def __init__(self,
                 n_feat):
        super(Fusion, self).__init__()
        # self.down1 = Downsample_block(in_chan, 32)
        # self.down2 = Downsample_block(32, 64)
        # self.down3 = Downsample_block(64, 128)
        # # self.down4 = Downsample_block(256, 512)
        # self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # # self.up4 = Upsample_block(1024, 512)
        # self.up3 = Upsample_block(256, 128)
        # self.up2 = Upsample_block(128, 64)
        # self.up1 = Upsample_block(64, 32)
        # self.outconv = nn.Conv2d(64, 64, 1)
        # self.outconv1 = nn.Conv2d(64, 32, 1)
        # self.outconv2 = nn.Conv2d(32, 16, 1)
        # self.outconv3 = nn.Conv2d(16, 1, 1)
        # self.outconvp1 = nn.Conv2d(32, out_chan, 1)
        # self.outconvm1 = nn.Conv2d(32, out_chan, 1)
        # self.flow = SqueezeBodyEdge(out_chan)
        self.fusion = FRNet(32, 8)

        self.segmodel = Segmentation()
        # self.decomodel = CUNet()
        self.decomodel = GPPNN(1,1,n_feat)

    def forward(self, x, y, seg_32):
        foutput, f_unique_x, f_unique_y, feature_x, feature_y, fcommon = self.decomodel(x, y)

        # output = self.fusion(torch.cat([x, y], dim=1))

        # z_rec, rec_u_1, rec_z_1, rec_v_1, rec_z_32 = self.decomodel(x_1, x_2)
        # output = self.outconv(torch.cat([fcommon, seg_32], dim=1))
        # output = self.outconv1(output)
        
        # output = self.outconv2(output)
        # output = self.outconv3(output)
        output, x_dec, x_seg, com_feature, wegiht_matrix = self.fusion(fcommon, seg_32)
        output = output + f_unique_x + f_unique_y

        # return output,  x_dec, x_seg, com_feature, wegiht_matrix
        return output, f_unique_x, f_unique_y, feature_x, feature_y
        # return output
        # return x1_r, x1_e, x2_r, x2_e, output


def feature_save(tensor,name,i):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    tensor = torch.mean(tensor,dim=1)
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    inp = inp.squeeze(2)
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    if not os.path.exists(name):
        os.makedirs(name)
    # for i in range(tensor.shape[1]):
    #     inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
    #     inp = np.clip(inp,0,1)
    # # inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
    #
    #     cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(name + '/' + str(i) + '.png', inp)


class EdgeBlock(nn.Module):
    def __init__(self, channelin, channelout):
        super(EdgeBlock, self).__init__()
        self.process = nn.Conv2d(channelin,channelout,3,1,1)
        self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
            nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
        self.CDC = cdcconv(channelin, channelout)

    def forward(self,x):

        x = self.process(x)
        out = self.Res(x) + self.CDC(x)

        return out

class FeatureExtract(nn.Module):
    def __init__(self, channelin, channelout):
        super(FeatureExtract, self).__init__()
        self.conv = nn.Conv2d(channelin,channelout,1,1,0)
        self.block1 = EdgeBlock(channelout,channelout)
        self.block2 = EdgeBlock(channelout, channelout)

    def forward(self,x):
        xf = self.conv(x)
        xf1 = self.block1(xf)
        xf2 = self.block2(xf1)

        return xf2

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        # self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc1_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # 
        # self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc1_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        self.fc1_rgb3 = nn.Linear(channels * 1 * 40 * 40, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 40 * 40, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 40 * 40, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 40 * 40, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        # if rgb_feat.shape[2] == 16:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 16 * 16)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 16 * 16)
        #
        #     mu_rgb = self.fc1_rgb1(rgb_feat)
        #     logvar_rgb = self.fc2_rgb1(rgb_feat)
        #     mu_depth = self.fc1_depth1(depth_feat)
        #     logvar_depth = self.fc2_depth1(depth_feat)
        # elif rgb_feat.shape[2] == 22:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 22 * 22)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 22 * 22)
        #     mu_rgb = self.fc1_rgb2(rgb_feat)
        #     logvar_rgb = self.fc2_rgb2(rgb_feat)
        #     mu_depth = self.fc1_depth2(depth_feat)
        #     logvar_depth = self.fc2_depth2(depth_feat)
        # else:
        rgb_feat = rgb_feat.view(-1, self.channel * 1 * 40 * 40)
        depth_feat = depth_feat.view(-1, self.channel * 1 * 40 * 40)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss

class GPPNN(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat):
        super(GPPNN, self).__init__()
        self.extract_pan = FeatureExtract(pan_channels,n_feat//2)
        self.extract_ms = FeatureExtract(ms_channels,n_feat//2)

        self.fus1 = FeatureExtract(n_feat, n_feat//2)
        self.fus2 = FeatureExtract(n_feat//2, 1)
        # self.mulfuse_pan = Multual_fuse(n_feat//2,n_feat//2)
        # self.mulfuse_ms = Multual_fuse(n_feat // 2, n_feat // 2)

        # self.interact = FeatureInteract(n_feat, n_feat//2)
        # self.refine = Refine(n_feat, ms_channels)

    def forward(self, x, y):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]

        _, _, m, n = x.shape
        

        feature_x = self.extract_pan(x)
        feature_y = self.extract_ms(y)

        # feature_save(feature_x, '/home/jieh/Projects/PAN_Sharp/PansharpingMul/GPPNN/training/logs/GPPNN2/panf')
        # feature_save(feature_y, '/home/jieh/Projects/PAN_Sharp/PansharpingMul/GPPNN/training/logs/GPPNN2/mHRf')

        finput = torch.cat([feature_x, feature_y], dim=1)
        fcommon = self.fus1(finput)
        unique_x = feature_x - fcommon   #channel of fcommon, unique_x, unique_y is "n_feat//2" 
        unique_y = feature_y - fcommon   #channel of foutput, f_unique_x, f_unique_y is "1"  
        f_unique_x = self.fus2(unique_x)
        f_unique_y = self.fus2(unique_y)
        foutput = self.fus2(fcommon)

        # HR = self.refine(fmid)+mHR

        return foutput, f_unique_x, f_unique_y, feature_x, feature_y, fcommon



if __name__ == '__main__':
    x_1 = torch.randn(1, 2, 160, 160).cuda()
    x_2 = torch.randn(1, 1, 160, 160).cuda()
    x_seg = torch.randn(1, 32, 160, 160).cuda()
    # print(x_emb.size())
    # frnet = FRNet(64,8)
    # test = Fusion(64).cuda()
    # decomnet= DecomNet()
    # segnet = SegNet(2,1)
    seg = Segmentation(2,1).cuda()
    # x_emb2 = frnet(x_emb)
    out = seg(x_1)
    # _, _, x1_16 = decomnet(x_1)
    # _, _, x2_16 = decomnet(x_2)
    # start = time.time()
    # output, f_unique_x, f_unique_y, feature_x, feature_y,_,_ = test(x_1, x_2, x_seg)
    # end = time.time()
    # output = frnet(torch.cat([x1_16, x2_16, x_seg], dim=1))
    # print('Running Time : ', str(end - start) + 's')
    # print(x_16.size())
    # print(x_32.size())

    # from thop import profile
    # # input_tensor = torch.rand(1, 1, 224, 224)   # 模型输入的形状,batch_size=1
    # flops, params = profile(test, inputs=(x_1, x_2, x_seg))
    # print("flops:", flops/1e9, "G")
    # print("params:", params/1e6, "M")
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # total = sum([param.nelement() for param in test.parameters()])
    # print('  + Number of params: %.5fM' % (total/ 1e6))

