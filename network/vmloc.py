import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
from torchvision import transforms, models
import torch.distributions as dist
import math
from tools.utils import repeat
from tools.utils import Constants

class MVFeature_Ext(nn.Module):
    def __init__(self, n_latents, is_train):
        super(MVFeature_Ext, self).__init__()
        self._pz_params = [torch.zeros(1, n_latents),torch.zeros(1, n_latents)]
        self.image_encoder = Image_Feature_Extractor(n_latents, is_train)
        self.depth_encoders = Depth_Feature_Extractor(n_latents, is_train)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.pz = dist.Normal
        self.qz_x = dist.Normal
    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
    def forward(self, image=None,depth=None,K=1,opt=None, epoch=None):
        self.qz_x_para = self.infer(image, depth)
        # Introducing the pretraining concept
        if epoch >= opt.train_mean:
            qz_x = self.qz_x(*self.qz_x_para)
            z = qz_x.rsample(torch.Size([K]))
        else:
            z = repeat(self.qz_x_para[0], K)
        return z, self.qz_x_para
    def infer(self, image=None, depth=None):
        if image is not None:
            batch_size = len(image)
        else:
            batch_size = len(depth)
        use_cuda = next(self.parameters()).is_cuda
        mu, logvar = prior_expert((1, batch_size, self.n_latents),use_cuda=use_cuda)
        if image is not None:
            image_mu, image_logvar = self.image_encoder(image)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)
        if depth is not None:
            depth_mu, depth_logvar = self.depth_encoders(depth)
            mu = torch.cat((mu, depth_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, depth_logvar.unsqueeze(0)), dim=0)
        mu, var = self.experts(mu, logvar)
        var_param = 1e-3
        return mu, var_param*var

class Depth_Feature_Extractor(nn.Module):
    def __init__(self, feat_dim, is_train=True):
        super(Depth_Feature_Extractor, self).__init__()
        self.feature_extractor = models.resnet34(pretrained=True)
        #self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feat_dim = feat_dim
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.fc1 = nn.Linear(fe_out_planes, feat_dim)
        self.fc2 = nn.Linear(fe_out_planes, feat_dim)
        if is_train == True:
            init_modules = [self.fc1, self.fc2]
            for m in init_modules:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[1:])
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        mu = self.fc1(x)
        var = self.fc2(x)
        return mu, torch.log(F.softplus(var) + Constants.eta) # Mu LogVariance

class Image_Feature_Extractor(nn.Module):
    def __init__(self, feat_dim, is_train=True):
        super(Image_Feature_Extractor, self).__init__()
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feat_dim = feat_dim
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.fc1 = nn.Linear(fe_out_planes, feat_dim)
        self.fc2 = nn.Linear(fe_out_planes, feat_dim)
        if is_train == True:
            init_modules = [self.fc1, self.fc2]
            for m in init_modules:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[1:])
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        mu = self.fc1(x)
        var = self.fc2(x)
        return mu, torch.log(F.softplus(var) + Constants.eta) # Mu LogVariance

class ProductOfExperts(nn.Module):
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        return pd_mu, pd_var

def prior_expert(size, use_cuda=False):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

def dreg(kl,loss_tmp, zs, args):
    lw = -loss_tmp+kl
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return -(grad_wt * lw).sum()

def Kld(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD)
    return KLD

def kl_divergence(d1, d2, K=100):
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def iwae(kl,loss_tmp,args):
    lw = -loss_tmp+kl
    loss = log_mean_exp(lw).sum()
    return (-loss)

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))