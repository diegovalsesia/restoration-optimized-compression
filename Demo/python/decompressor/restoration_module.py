import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as stx
import numbers

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np



class CatRestormer(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(CatRestormer, self).__init__()

        ckpt = torch.load('srx8.pkl', map_location='cuda')
        self.up_net = SYESRX8NetS(36)
        self.up_net.load_state_dict(ckpt)
        for param in self.up_net.parameters():
            param.requires_grad = False

        self.resotrmer = Restormer(
            inp_channels=inp_channels, 
            out_channels=out_channels, 
            dim = dim,
            num_blocks = num_blocks, 
            num_refinement_blocks = num_refinement_blocks,
            heads = heads,
            ffn_expansion_factor = ffn_expansion_factor,
            bias = bias,
            LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
            dual_pixel_task = dual_pixel_task)
        
    def forward(self,x,dep):
        dep = self.up_net(dep)
        b,c,h,w = x.size() 
        dep = F.interpolate(dep, size=(h,w))
        inp = torch.cat((x,dep),dim=1)
        return self.resotrmer(inp)
    


class PrePyramidL1S(nn.Module):
    def __init__(self, num_feat):
        super(PrePyramidL1S, self).__init__()
        self.conv_first = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.resblock = ResBlockS(num_feat=num_feat)

    def forward(self, x):
        feat_l1 = self.conv_first(x)
        feat_l1 = self.resblock(feat_l1)
        return feat_l1



class PrePyramidL2S(nn.Module):
    def __init__(self, num_feat):
        super(PrePyramidL2S, self).__init__()
        self.conv_first = nn.Conv2d(1, num_feat, 3, 1, 1)
        self.resblock = ResBlockS(num_feat=num_feat)

    def forward(self, x):
        feat_l2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        feat_l2 = self.conv_first(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        _, _, h, w = x.size()
        feat_l2 = nn.Upsample((h, w), mode='bilinear', align_corners=False)(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        return feat_l2


class SYESRX8NetS(nn.Module):
    def __init__(self, channels):
        super(SYESRX8NetS, self).__init__()
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.headpre = AdditionFusionS(PrePyramidL1S(1), PrePyramidL2S(1), 1)
        self.resblock = ResBlockS(num_feat=1)
        self.head = QuadraticConnectionUnitS(
            nn.Sequential(
                nn.Conv2d(1, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            nn.Conv2d(1, channels, 5, 1, 2),
            channels
        )
        self.body = QuadraticConnectionUnitS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, ),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels,  channels, 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        self.tail = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 1, 3, 1, 1)
        )

    def forward(self, x):
        inp = x
        x = self.headpre(x)
        x = self.resblock(x)
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        base = F.interpolate(inp, scale_factor=8, mode='bilinear', align_corners=False)
        x = self.tail(x) + base
        return x 
    




class QuadraticConnectionUnitS(nn.Module):
    def __init__(self, block1, block2, channels):
        super(QuadraticConnectionUnitS, self).__init__()
        self.block1 = block1
        self.block2 = block2
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        return self.block1(x) * self.block2(x) + self.bias

class AdditionFusionS(nn.Module):
    def __init__(self, addend1, addend2, channels):
        super(AdditionFusionS, self).__init__()
        self.addend1 = addend1
        self.addend2 = addend2
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        return self.addend1(x) + self.addend2(x) + self.bias



class ResBlockS(nn.Module):
    def __init__(self, num_feat=4):
        super(ResBlockS, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out
    



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        if LayerNorm_type =='BiasFree':
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

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

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
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        fig = plt.figure()
        # dep_fig = torch.sum(inp_dec_level3,dim=1)
        # dep_fig = dep_fig.cpu().squeeze(0).numpy()
        # dep_fig = (dep_fig - np.min(dep_fig))/(np.max(dep_fig) -np.min(dep_fig))
        # plt.imshow(dep_fig)
        # fig.savefig("./depth_tensor/nodepth_level3.png")
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # fig = plt.figure()
        # dep_fig = torch.sum(inp_dec_level2,dim=1)
        # dep_fig = dep_fig.cpu().squeeze(0).numpy()
        # dep_fig = (dep_fig - np.min(dep_fig))/(np.max(dep_fig) -np.min(dep_fig))
        # plt.imshow(dep_fig)
        # fig.savefig("./depth_tensor/nodepth_level2.png")
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # fig = plt.figure()
        # dep_fig = torch.sum(inp_dec_level1,dim=1)
        # print(dep_fig.size())
        # dep_fig = dep_fig.cpu().squeeze(0).numpy()
        # dep_fig = (dep_fig - np.min(dep_fig))/(np.max(dep_fig) -np.min(dep_fig))
        # plt.imshow(dep_fig)
        # fig.savefig("./depth_tensor/nodepth_level1.png")
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
        ####this part is revised for concat image, but will not effect the performance
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,:3,:,:]


        return out_dec_level1