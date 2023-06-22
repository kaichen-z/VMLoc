import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
import sys

import time
import os.path as osp
import numpy as np
from PIL import ImageFile
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
from data.dataloaders import SevenScenes, RobotCar
from tools.options import *
from tools.options import Options
from tools.utils import AtLocCriterion, multi_AtLocCriterion, AverageMeter, Logger
from tools.utils import quaternion_angular_error, qexp, load_state_dict, written
from network.atloc import AtLoc
from network.ae import AEFeature_Ext
from network.vmloc import MVFeature_Ext, dreg, iwae
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from tqdm import tqdm
import pynvml
from torchvision import utils as UTILS
import matplotlib.pyplot as plt

def main():
    '''------------------------Config------------------------'''
    opt = Options().parse()
    cuda = torch.cuda.is_available()
    device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"
    logfile = osp.join(opt.runs_dir, 'log.txt')
    stdout = Logger(logfile)
    print(device)
    print('Logging to {:s}'.format(logfile))
    sys.stdout = stdout
    '''------------------------Random Seed------------------------'''
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    '''------------------------Modelling Definition------------------------'''
    if opt.mode == "concatenate": # Concatenatings 
        feature_extractor = AEFeature_Ext(n_latents=opt.dim, is_train=True)
    elif opt.mode == "vmloc": # Dreg for both image and depth
        feature_extractor = MVFeature_Ext(n_latents=opt.dim, is_train=True)
    atloc = AtLoc(opt, feature_extractor, droprate=opt.train_dropout, initialize=True)
    model = atloc
    '''------------------------Model Loading------------------------'''
    if opt.weights != None:
        print('Loading the model {}'.format(opt.weights))
        weights_filename = osp.expanduser(opt.weights)
        if osp.isfile(weights_filename):
            checkpoint = torch.load(weights_filename, map_location=lambda storage, loc: storage.cuda(0))
            load_state_dict(model, checkpoint['model_state_dict'])
            print('Loaded weights from {:s}'.format(weights_filename))
        else:
            print('Could not load weights from {:s}'.format(weights_filename))
            sys.exit(-1)
    '''--------------------Transformation_Image_Test----------------------'''
    stats_file = osp.join(opt.data_dir, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    tforms = [transforms.Resize(opt.cropsize)]
    tforms.append(transforms.CenterCrop(opt.cropsize))
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
    image_transform_val = transforms.Compose(tforms)
    '''-------------------Transformation_Depth----------------'''
    stats_file = osp.join(opt.data_dir, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    tforms = [transforms.Resize(opt.cropsize)]
    tforms.append(transforms.CenterCrop(opt.cropsize))
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
    depth_transform_val = transforms.Compose(tforms)
    '''--------------------Transformation_Target----------------------'''
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
    '''--------------------Data_Loading----------------------'''
    val_kwargs = dict(scene=opt.scene, data_path=opt.data_dir, image_transform=image_transform_val, depth_transform=depth_transform_val, target_transform=target_transform, seed=opt.seed)
    if opt.dataset == '7Scenes':
        val_set = SevenScenes(train=False, **val_kwargs)
    elif opt.dataset == 'RobotCar':
        val_set = RobotCar_Noise(train=False, length=length, mask=mask, **val_kwargs)
    else:
        raise NotImplementedError
    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
    '''--------------------Model_To_Cuda----------------------'''
    model = model.to(device)
    '''--------------------Validation_Beginning----------------------'''
    experiment_name = opt.exp_name
    val_loss = AverageMeter()
    model.eval()
    end = time.time()
    val_batch_time = AverageMeter()
    val_data_time = AverageMeter()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
    meminfo_start = pynvml.nvmlDeviceGetMemoryInfo(handle)
    val_batch_storage = AverageMeter()
    
    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error
    L = len(val_set)
    pose_stats_file = osp.join(opt.data_dir, opt.scene, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
    pred_poses = np.zeros((L, 7))  # store all predicted poses
    targ_poses = np.zeros((L, 7))  # store all target poses

    '''----------------------------'''
    val_loader =tqdm(val_loader)
    for batch_idx, (val_rgb_var, val_depth_var, val_target) in enumerate(val_loader):
        val_data_time.update(time.time() - end)
        val_rgb_var = Variable(val_rgb_var, requires_grad=False).to(device)
        val_depth_var = Variable(val_depth_var, requires_grad=False).to(device)
        val_target_var = Variable(val_target, requires_grad=False).to(device)
        with torch.set_grad_enabled(False):
            if opt.mode == "concatenate":
                val_output = model(image=val_rgb_var, depth=val_depth_var, mode='val')
            elif opt.mode == "vmloc":
                val_output, _, _, _ = model(image=val_rgb_var, depth=val_depth_var, K=opt.K, epoch=10000, mode='val')
                val_output = val_output.view(val_rgb_var.size(0),opt.K,-1)
                val_output = torch.mean(val_output,dim=1)
                val_output = val_output.view(1, val_rgb_var.size(0), -1)
        val_batch_time.update(time.time() - end)
        val_batch_storage.update(pynvml.nvmlDeviceGetMemoryInfo(handle).used - meminfo_start.used)
        '''----------------------------'''
        s = val_output.size()
        output = val_output.cpu().data.numpy().reshape((-1, s[-1]))
        target = val_target_var.cpu().data.numpy().reshape((-1, s[-1]))
        # normalize the predicted quaternions
        q = [qexp(p[3:]) for p in output]
        output = np.hstack((output[:, :3], np.asarray(q)))
        q = [qexp(p[3:]) for p in target]
        target = np.hstack((target[:, :3], np.asarray(q)))
        # un-normalize the predicted and target translations
        output[:, :3] = (output[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m
        # take the middle prediction
        pred_poses[batch_idx, :] = output[int(len(output) / 2)]
        targ_poses[batch_idx, :] = target[int(len(target) / 2)]
        end = time.time()
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
    print('Error in translation: median {:3.2f} m,  mean {:3.2f} m \nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))
    print('------------Time consuming: {:f}---Storaing: {:f}---------'.format(val_batch_time.avg, val_batch_storage.avg))

    fig = plt.figure()
    real_pose = (pred_poses[:, :3] - pose_m) / pose_s
    gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
    plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
    plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
    plt.show(block=True)
    image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
    fig.savefig(image_filename)

if __name__ == "__main__":
    main()
