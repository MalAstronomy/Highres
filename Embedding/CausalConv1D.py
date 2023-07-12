import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Conv1d 

class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, A=False, **kwargs):
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A
        
        self.padding = (kernel_size - 1) * dilation + A * 1

        # module:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride,
                                      padding=0,
                                      dilation=dilation,
                                      **kwargs)


    def forward(self, x):
        # x = torch.nn.functional.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)

        if self.A:
            return conv1d_out[:, :, : -1]
        else:
            return conv1d_out
        

    

class CausalConvLayers(nn.Module):

    def __init__(self, in_channels, out_channels, MM):
        super(CausalConvLayers, self).__init__()

        self.layers= nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=MM, dilation=1, kernel_size=32, A=True, bias = True), 
            nn.LeakyReLU(), 
            CausalConv1d(in_channels=MM, out_channels=MM, dilation=2, kernel_size=32, A=False, bias = True), 
            nn.LeakyReLU(), 
            CausalConv1d(in_channels=MM, out_channels=MM, dilation=2, kernel_size=32, A=False, bias = True), 
            nn.LeakyReLU(), 
            CausalConv1d(in_channels=MM, out_channels=MM*2, dilation=2, kernel_size=32, A=False, bias = True), 
            nn.LeakyReLU(), 
            Conv1d(in_channels=MM*2, out_channels=MM*4, kernel_size=64, stride= 32), 
            nn.LeakyReLU(),
            Conv1d(in_channels=MM*4, out_channels= out_channels, kernel_size=64, stride= 32), 
            nn.LeakyReLU(),
            )
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.layers(x)
        x = self.flatten(x)
        return x

