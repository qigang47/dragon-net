import torch
from torch import Tensor
from torch import nn
from model.utils import  PatchEmbed,RepMixer,ConvFFN
from model.attention import eca_layer
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRM_npy1 = np.load('model/SRM3_3.npy')
SRM_npy2 = np.load('model/SRM5_5.npy')

class pre_layer(nn.Module):
    def __init__(self):
        super(pre_layer, self).__init__()
        self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)
        srm_weights = torch.from_numpy(np.load("model/srm.npy")).float()
        gabor_weights = torch.from_numpy(np.load("model/gabor32.npy").reshape(32, 1, 5, 5)).float()
        self.conv1 = nn.Conv2d(1,30,5,1,2)
        self.conv1.weight = nn.Parameter(srm_weights,False)
        self.conv1.bias = nn.Parameter(torch.zeros(30))
        self.conv2 = nn.Conv2d(1,30,5,1,2)
        self.conv1.bias = nn.Parameter(torch.zeros(30))
        self.conv1.weight = nn.Parameter(srm_weights, True)

        self.conv3 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv4 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv3.weight = nn.Parameter(gabor_weights, False)
        self.conv3.bias = nn.Parameter(torch.zeros(32))
        self.conv4.weight = nn.Parameter(gabor_weights, True)
        self.conv4.bias = nn.Parameter(torch.zeros(32))
        self.eca=eca_layer(62)
    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return(self.tlu (self.eca ( torch.cat([x1,x3],1))))


class Srnet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.pre = pre_layer()
        self.batch_norm = nn.BatchNorm2d(62)
        self.conv3 = nn.Conv2d(62,32, 3, 1, 1)
        self.bn =nn.BatchNorm2d(32)
        self.prelu =nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.type1s = nn.Sequential( PatchEmbed(stride=2, in_channels=32, embed_dim=64),
            RepMixer(dim=64),
            ConvFFN(64,128,64),)
        self.type2s = nn.Sequential(
            PatchEmbed(stride=2, in_channels=64, embed_dim=128),
            RepMixer(dim=128),
            ConvFFN(128,256,128),
        )
        self.type3s = nn.Sequential(
            PatchEmbed(stride=1, in_channels=128,  embed_dim=256),

        )

        self.classfier = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128 , 2)
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
    def forward(self, x: Tensor) -> Tensor:
        x =self.pre(x)

        out= self.batch_norm(x)
        out= self.prelu1(out)
        out = self.conv3(out)
        out =self.bn(out)
        out =self.prelu(out)
        out = self.type1s(out)
        out = self.type2s(out)
        out = self.type3s(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        x = self.classfier(out)
        return torch.sigmoid(x)


if __name__ == "__main__":
    image = torch.randn((2, 1, 256, 256))
    net = Srnet()
    print(net(image).shape)
