import os
import torch
from torch import nn
import transforms3d.quaternions as txq
import numpy as np
import sys
from torchvision.datasets.folder import default_loader
from collections import OrderedDict
import os.path as osp

class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    def forward(self, pred, targ):
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq
        return loss

class multi_AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(multi_AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    def forward(self, pred, targ):
        s0 = pred.size()
        pred = pred.view(s0[1],s0[0],*s0[2:])
        s = pred.size()
        abs_loss = torch.exp(-self.sax) *torch.mean(torch.abs(repeat(targ, s[0]).view(s[0],s[1],-1)[:,:,:3]-pred.view(s[0],s[1],-1)[:,:,:3]),dim=2) + self.sax + \
                   torch.exp(-self.saq) *torch.mean(torch.abs(repeat(targ, s[0]).view(s[0],s[1],-1)[:, :, 3:]-pred.view(s[0],s[1], -1)[:, :, 3:]), dim=2) + self.saq
        abs_loss = abs_loss.permute(1,0)
        return abs_loss

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img

def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

def calc_vos_simple(poses):
    vos = []
    for p in poses:
        pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos

def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta
    
def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out

def repeat(tensor, K):
    tensor_new = tensor.unsqueeze(0)
    tensor_final = tensor_new
    for i in range(K-1):
        tensor_final = torch.cat((tensor_final, tensor_new),dim=0)
    return tensor_final

def load_state_dict(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]
  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

class Constants(object):
    eta = 1e-6

def written(opt, contain):
    f = open(osp.join(opt.record_dir, "record_{}_{}.txt".format(opt.model, opt.mode)), "a")
    f.write(contain)
    f.close()