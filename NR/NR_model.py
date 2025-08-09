import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, create_model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class SigLIP2_384(nn.Module):
    def __init__(self):
        super(SigLIP2_384, self).__init__()

        model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2-384', pretrained=False)

        # spatial quality analyzer
        self.feature_extraction = model.visual

        # quality regressor
        self.quality = self.quality_regression(768, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)

        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x = self.quality(x)
        
        # x size: batch x frames
        x = x.view(x_size[0], x_size[1])

        # x size: batch
        x = torch.mean(x, dim = 1)
            
        return x



class SigLIP2_384_multi_dataset(nn.Module):
    def __init__(self):
        super(SigLIP2_384_multi_dataset, self).__init__()

        # model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2-384', pretrained=False)
        model = create_model('ViT-B-16-SigLIP2-384', pretrained=False)

        # spatial quality analyzer
        self.feature_extraction = model.visual

        # quality regressor
        self.quality1 = self.quality_regression(768, 128, 1)
        self.quality2 = self.quality_regression(768, 128, 1)
        self.quality3 = self.quality_regression(768, 128, 1)
        self.quality4 = self.quality_regression(768, 128, 1)
        self.quality5 = self.quality_regression(768, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        # x size: batch x frames x 3 x height x width
        x_size = x.shape
        
        # x size: (batch * frames) x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
            
        x = self.feature_extraction(x)

        # x size: (batch * frames) x channel
        x = torch.flatten(x, 1)

        # x size: (batch * frames)
        x1 = self.quality1(x)        
        # x size: batch x frames
        x1 = x1.view(x_size[0], x_size[1])
        # x size: batch
        x1 = torch.mean(x1, dim = 1)

        # x size: (batch * frames)
        x2 = self.quality2(x)        
        # x size: batch x frames
        x2 = x2.view(x_size[0], x_size[1])
        # x size: batch
        x2 = torch.mean(x2, dim = 1)

        # x size: (batch * frames)
        x3 = self.quality3(x)        
        # x size: batch x frames
        x3 = x3.view(x_size[0], x_size[1])
        # x size: batch
        x3 = torch.mean(x3, dim = 1)

        # x size: (batch * frames)
        x4 = self.quality4(x)        
        # x size: batch x frames
        x4 = x4.view(x_size[0], x_size[1])
        # x size: batch
        x4 = torch.mean(x4, dim = 1)

        # x size: (batch * frames)
        x5 = self.quality5(x)        
        # x size: batch x frames
        x5 = x5.view(x_size[0], x_size[1])
        # x size: batch
        x5 = torch.mean(x5, dim = 1)
            
        return x1, x2, x3, x4, x5