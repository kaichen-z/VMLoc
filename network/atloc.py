import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)
        self.W = nn.Linear(in_channels // 8, in_channels)
    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)
        g_x = self.g(x).view(batch_size, out_channels // 8, 1)
        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z

class AtLoc(nn.Module):
    def __init__(self, opt, feature_extractor, droprate=0.5, initialize=True):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        feat_dim = opt.dim
        self.opt = opt
        if opt.mode == "concatenate":
            self.feature_extractor = feature_extractor
            self.att = AttentionBlock(feat_dim)
            self.fc_xyz1 = nn.Linear(feat_dim, 3)
            self.fc_wpqr1 = nn.Linear(feat_dim, 3)
        elif opt.mode == "vmloc":
            self.feature_extractor = feature_extractor
            self.att = AttentionBlock(feat_dim)
            self.fc_xyz1 = nn.Linear(feat_dim, 3)
            self.fc_wpqr1 = nn.Linear(feat_dim, 3)

        if initialize:
            if opt.mode == "concatenate":
                init_modules = [self.fc_xyz1, self.fc_wpqr1]
            elif opt.mode == "vmloc":
                init_modules = [self.fc_xyz1, self.fc_wpqr1]
        else:
            init_modules = self.modules()
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    
    def forward(self, image=None, depth=None, K = None, epoch=None, mode='train'):
        if self.opt.mode == "concatenate":
            feature = self.feature_extractor(image, depth)
            x = F.tanh(feature)
            x = self.att(x.view(x.size(0), -1))
            if self.droprate > 0 and mode == 'train':
                x = F.dropout(x, p=self.droprate)
            xyz = self.fc_xyz1(x)
            wpqr = self.fc_wpqr1(x)
            return torch.cat((xyz, wpqr), 1)
        elif self.opt.mode == "vmloc":
            feature, qz_x_para = self.feature_extractor(image=image, depth=depth, K=K, opt=self.opt, epoch=epoch) # Nor: (K,B,:); Plus:(K,BS,:)
            feature_size = feature.size()
            feature = feature.view(-1, *feature_size[2:]) # Nor: (KB,:); Plus:(KBS,:)
            x = F.tanh(feature)
            x = self.att(x.view(x.size(0), -1))
            if self.droprate > 0 and mode == 'train':
                x = F.dropout(x, p=self.droprate)
            xyz = self.fc_xyz1(x)
            wpqr = self.fc_wpqr1(x)
            kl1 = self.feature_extractor.pz(self.feature_extractor.pz_params[0].cuda(), self.feature_extractor.pz_params[1].cuda()).log_prob(feature.view(*feature_size)).sum(-1).view(-1) # P(Z) KB
            kl2 = self.feature_extractor.qz_x(qz_x_para[0], qz_x_para[1]).log_prob(feature.view(*feature_size)).sum(-1).view(-1) # KB Q(Z|X)
            return torch.cat((xyz, wpqr), 1), feature, qz_x_para, [kl1,kl2]



