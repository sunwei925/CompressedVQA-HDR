import torch
from torchvision import models
import torch.nn as nn
from torchvision.ops.misc import Permute


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Swin_b_FR_multi_stage(torch.nn.Module):
    def __init__(self, pretrained = True):        
        super(Swin_b_FR_multi_stage, self).__init__()

        if pretrained == True:
            swin_b = models.swin_b(weights='Swin_B_Weights.DEFAULT')
        else:
            swin_b = models.swin_b()


        self.feature_extraction_stage1 = torch.nn.Sequential()
        self.feature_extraction_stage1.add_module(str(0), swin_b.features[0])
        self.feature_extraction_stage1.add_module(str(1), swin_b.features[1])

        self.feature_extraction_stage2 = torch.nn.Sequential()
        self.feature_extraction_stage2.add_module(str(2), swin_b.features[2])
        self.feature_extraction_stage2.add_module(str(3), swin_b.features[3])

        self.feature_extraction_stage3 = torch.nn.Sequential()
        self.feature_extraction_stage3.add_module(str(4), swin_b.features[4])
        self.feature_extraction_stage3.add_module(str(5), swin_b.features[5])

        self.feature_extraction_stage4 = torch.nn.Sequential()
        self.feature_extraction_stage4.add_module(str(6), swin_b.features[6])
        self.feature_extraction_stage4.add_module(str(7), swin_b.features[7])

        
        self.norm = swin_b.norm
        self.permute = Permute([0, 3, 1, 2])
        

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,128,256,512,1024]

        self.quality = self.quality_regression(sum(self.chns)*2,128,1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward_once(self, h):
        # h = (x-self.mean)/self.std
        x = h*self.std + self.mean
        h = self.feature_extraction_stage1(h)
        h_1 = self.permute(h)
        h = self.feature_extraction_stage2(h)
        h_2 = self.permute(h)
        h = self.feature_extraction_stage3(h)
        h_3 = self.permute(h)
        h = self.feature_extraction_stage4(h)
        h_4 = self.permute(h)
        return [x,h_1, h_2, h_3, h_4]

    def forward(self, x, y):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        y = y.view(-1, x_size[2], x_size[3], x_size[4])

        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)

        c1 = 1e-6
        c2 = 1e-6

        S = []

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            S.append(S1)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            S.append(S2)

        feats = torch.cat(S, dim = 1).squeeze()
        x = self.quality(feats)

        # x: batch x frames
        x = x.view(x_size[0],x_size[1])
        # x: batch x 1
        x = torch.mean(x, dim = 1)
            
        return x