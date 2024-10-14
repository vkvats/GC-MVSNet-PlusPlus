import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


## -------------------------Dense3D CostReg Net (modified) -------------------------------------

class DenseBlock3d(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, dropout=True):
        super().__init__() 
        self.drop = dropout
        self.layers = nn.ModuleList([Conv3d(in_channels + i*growth_rate, growth_rate, 
                                           padding=1) for i in range(n_layers)])
        self.dropoutlayers = nn.ModuleList([nn.Dropout3d(0.2) for i in range(n_layers)])

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            out = layer(x) 
            if self.drop: 
                out = self.dropoutlayers[idx](out)
            x = torch.cat([x, out], 1) # 1 = channel axis 
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_channels, dropout=True):
        super().__init__()
        self.drop = dropout
        self.down1 = Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(2)
        if self.drop:
            self.dropout = nn.Dropout3d(0.2) 

    def forward(self, x):
        x = self.down1(x)
        if self.drop: 
            x = self.dropout(x)
        x = self.maxpool(x)
#         print(f"x after MP:{x.shape}")
        return x
    
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = Deconv3d(in_channels=in_channels, 
                                  out_channels=out_channels,
                                  stride=2, 
                                  padding=1,
                                  output_padding=1)

    def forward(self, x, skip):
        output = self.convTrans(x)      
#         print(f"transition UP output:{output.shape}")
        return output if skip is None else output + skip
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        ## growth_rate * n_layers = in_channels for bottleneck
        self.layers = nn.ModuleList([Conv3d(in_channels + i*growth_rate, growth_rate, 
                                           padding=1) for i in range(n_layers)])

    def forward(self, x):
        new_features = []
        #we pass all previous activations into each dense layer normally
        #But we only store each dense layer's output in the new_features array
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
            new_features.append(out)

        return torch.cat(new_features,1)
    
class Dense3DUNet(nn.Module):
    def __init__(self, in_channels=1, 
                 down_blocks=(2,2,2),
                 up_blocks=(2,2,2), 
                 bottleneck_layers=4,
                 growth_rate=(4,4,4), 
                 base_channel=8):
        super().__init__()        
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        self.finalconv = Conv3d(base_channel * 2, base_channel , padding=1)
        self.prob = nn.Conv3d(base_channel, 1, 3, stride=1, padding=1, bias=False)
        ## First conv layer to get base number of channels        
        self.add_module('firstconv', Conv3d(in_channels=in_channels, 
                                               out_channels=base_channel, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=1))
        cur_channels_count = base_channel

        ## Downsampling path 
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock3d(cur_channels_count, growth_rate[i], down_blocks[i]))
            cur_channels_count += (growth_rate[i]*down_blocks[i])
            
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        ## bottleneck growthrate = in_channels//bottlenect_layer
        bottleneck_GR = cur_channels_count//bottleneck_layers
        self.add_module('bottleneck',Bottleneck(in_channels=cur_channels_count,
                                                growth_rate=bottleneck_GR, 
                                                n_layers=bottleneck_layers))
        prev_block_channels = bottleneck_GR*bottleneck_layers
        cur_channels_count += prev_block_channels

        ##   Upsampling path
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)):
            # transition up 
            out_channel = prev_block_channels if i==0 else prev_block_channels//2
            self.transUpBlocks.append(TransitionUp(prev_block_channels, out_channel))
            
            prev_block_channels = out_channel
            growth_rate = prev_block_channels//up_blocks[i]
            self.denseBlocksUp.append(Bottleneck(in_channels=prev_block_channels,
                                                growth_rate=growth_rate, 
                                                n_layers=up_blocks[i]))

                                        
        
        # Official init from torch repository
        for m in self.modules():
            #print('initializing conv3d and batchnorm3d weights')
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        skip_connections.append(out)
        
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
        skip = skip_connections.pop()
        out = skip + self.finalconv(out)
        out = self.prob(out)                                    
        return out
