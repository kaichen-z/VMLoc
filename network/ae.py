from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from torchvision import transforms, models

class AEFeature_Ext(nn.Module):
    def __init__(self, n_latents, is_train):
        super(AEFeature_Ext, self).__init__()
        self.image_encoder = Image_Feature_Extractor(n_latents, is_train)
        self.depth_encoder = Depth_Feature_Extractor(n_latents, is_train)
        self.n_latents = n_latents
    def forward(self, image=None, depth=None):
        if image is not None:
            image_feature = self.image_encoder(image)
        else: 
            image_feature = torch.zeros(int(depth.size(0)), int(self.n_latents/2)).cuda()
        if depth is not None:
            depth_feature = self.depth_encoder(depth)
        else:
            depth_feature = torch.zeros(int(image.size(0)), int(self.n_latents/2)).cuda()
        feature = torch.cat((image_feature, depth_feature), dim=1)
        return feature

class Depth_Feature_Extractor(nn.Module):
    def __init__(self, feat_dim, is_train=True):
        super(Depth_Feature_Extractor, self).__init__()
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feat_dim = feat_dim
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, int(feat_dim/2))
        if is_train == True:
            init_modules = [self.feature_extractor.fc]
            for m in init_modules:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[1:])
        x = self.feature_extractor(x)
        return x

class Image_Feature_Extractor(nn.Module):
    def __init__(self, feat_dim, is_train=True):
        super(Image_Feature_Extractor, self).__init__()
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feat_dim = feat_dim
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, int(feat_dim/2))
        if is_train == True:
            init_modules = [self.feature_extractor.fc]
            for m in init_modules:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[1:])
        x = self.feature_extractor(x)
        return x