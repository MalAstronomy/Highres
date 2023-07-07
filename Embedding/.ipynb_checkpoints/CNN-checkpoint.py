import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, n_channels_in=2, n_channels_out=128):
        super(CNN, self).__init__()
        
        ## Convolutional layers ##
        self.layer1 = self.ConvLayer(n_channels_in, 4, ksize_conv=8, strd_conv=4)
        self.layer2 = self.ConvLayer(4, 8, ksize_conv=8, strd_conv=4)
        self.layer3 = self.ConvLayer(8, 16, ksize_conv=8, strd_conv=4)
        self.layer4 = self.ConvLayer(16, 32, ksize_conv=8, strd_conv=4)
        
        ## Fully connected layers ##
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1504, 256),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(128, n_channels_out))
        self.fc = self.fc.to(torch.float32)
        
    def ConvLayer(self, nb_neurons_in, nb_neurons_out, ksize_conv=3, strd_conv=1, pad_conv=0, ksize_pool=3, strd_pool=1, pad_pool=0):
        '''
        Define a convolutional layer
        '''
        layer = nn.Sequential(
            nn.Conv1d(nb_neurons_in, nb_neurons_out, 
                      kernel_size=ksize_conv, stride=strd_conv, padding=pad_conv),
            #nn.BatchNorm1d(nb_neurons_out),
            nn.ELU())
            #nn.MaxPool1d(kernel_size=ksize_pool, stride=strd_pool, padding=pad_pool))
        return layer

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.layer1(x)
        print(out.size())
        out = self.layer2(out)
        print(out.size())
        out = self.layer3(out)
        print(out.size())
        #print(out.dtype)
        out = out.view(out.size(0), -1)#.to(torch.float32)  # Flatten for fully connected layers
        out = self.fc(out)
        print(out.size())
        out = out.to(torch.double)
        
        return out