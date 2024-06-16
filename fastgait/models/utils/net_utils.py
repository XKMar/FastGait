import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    r"""
        Residual opreation.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images.
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        return self.fn(x) + x
    
class SetBlock(nn.Module):
    r"""
        SetBlock: Extracting features of sequences using 2D convolution
        The detail information refer to https://arxiv.org/abs/1811.06186.
    """
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B T C H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        b, t, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(b, t, c, h ,w)

class BasicConv2d(nn.Module):
    r"""
        Basic Conv2d with LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B C H W]
                (B: Batch, C: Channel, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        out = self.conv(x)
        out = self.relu(out)
        return out
    
class FocalConv2d(nn.Module):
    r"""
        Focal Conv2d for each split feature separately.
        The detail information refer to https://ieeexplore.ieee.org/document/9156784.
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B C H W]
                (B: Batch, C: Channel, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z

class BasicConv3d(nn.Module):
    r"""
        Basic Conv3d with LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B C T H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        out = self.conv(x)
        out = self.relu(out)
        return out
    
class GLConv3d(nn.Module):
    r"""
        Global & Local Conv3d with LeakyReLU.
        The detail information refer to https://arxiv.org/abs/2011.01461.
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, fm_sign=False, **kwargs):
        super(GLConv3d, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign

        self.gl_conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.lo_conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B C T H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        global_feat = self.gl_conv3d(x)
        if self.halving == 0:
            local_feat = self.lo_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            local_feat = x.split(split_size, 3)
            local_feat = torch.cat([self.lo_conv3d(_) for _ in local_feat], 3)

        if not self.fm_sign:
            feat = self.relu(global_feat) + self.relu(local_feat)
        else:
            feat = self.relu(torch.cat([global_feat, local_feat], dim=3))

        return feat

class SATMixedConv3d(nn.Module):
    r"""
        Combined 3D convolution and Spatial & Temporal Mixed operations.

        Args:
            in_channels   (int): The input channels of 3D convolution.
            out_channels  (int): The output channels of 3D convolution.
        
        outputs:
            tensor: The features containing local motion patterns.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc_s = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False)
        self.fc_t = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False)

        self.theta_s = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), 
                            padding=(0,0,0), groups=in_channels, bias=False), nn.ReLU())
        self.theta_t = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), 
                            padding=(0,0,0), groups=in_channels, bias=False), nn.ReLU())

        self.tfc_s = nn.Conv3d(out_channels*2, out_channels, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.tfc_t = nn.Conv3d(out_channels*2, out_channels, kernel_size=(7,1,1), stride=(1,1,1), padding=(3,0,0), bias=False)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B C T H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """

        # theta value
        theta_s = self.theta_s(x)
        theta_t = self.theta_t(x)

        # magnitude value
        x_s = self.fc_s(x)
        x_t = self.fc_t(x)

        # mix stage along spatial and temporal dimension separately
        out_s = torch.cat([x_s*torch.cos(theta_s), x_s*torch.sin(theta_s)], dim=1)
        out_t = torch.cat([x_t*torch.cos(theta_t), x_t*torch.sin(theta_t)], dim=1)

        out_s = self.tfc_s(out_s)
        out_s = self.relu(out_s)

        out_t = self.tfc_t(out_t)
        out_t = self.relu(out_t)

        out = out_s + out_t
        out = self.relu(out)

        return out

class STMixedConv3d(nn.Module):
    r"""
        Combined 3D convolution and Spatial & Temporal Mixed operations.

        Args:
            in_channels   (int): The input channels of 3D convolution.
            out_channels  (int): The output channels of 3D convolution.
        
        outputs:
            tensor: The features containing local motion patterns.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc_st = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), 
                                padding=(0,0,0), bias=False)
        self.theta_st = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), 
                                padding=(0,0,0), groups=in_channels, bias=False), nn.ReLU())

        self.tfc_st = nn.Conv3d(out_channels*2, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        r"""
            Forward function.
            Args:
                >>> Inputs: x (Tensor), Input images. [B C T H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """

        # theta and magnitude value
        x_st = self.fc_st(x)
        theta_st = self.theta_st(x)

        # mix stage according to the conv field
        out = torch.cat([x_st*torch.cos(theta_st), x_st*torch.sin(theta_st)], dim=1)
        out = self.tfc_st(out)
        out = self.relu(out)

        return out