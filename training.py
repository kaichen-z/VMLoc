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
import pynvml

def main():
    '''------------------------Config------------------------'''
    opt = Options().parse()
    cuda = torch.cuda.is_available()
    device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"
    if not os.path.exists(opt.runs_dir):
        os.makedirs(opt.runs_dir)
    if not os.path.exists(opt.models_dir):
        os.makedirs(opt.models_dir)       
    logfile = osp.join(opt.runs_dir, 'log.txt')
    stdout = Logger(logfile)
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
        train_criterion = nn.DataParallel(AtLocCriterion(saq=opt.beta, learn_beta=True))
        val_criterion = nn.DataParallel(AtLocCriterion())
    elif opt.mode == "vmloc": # Dreg for both image and depth
        feature_extractor = MVFeature_Ext(n_latents=opt.dim, is_train=True)
        train_criterion = nn.DataParallel(multi_AtLocCriterion(saq=opt.beta, learn_beta=True))
        val_criterion = nn.DataParallel(multi_AtLocCriterion())
    atloc = AtLoc(opt, feature_extractor, droprate=opt.train_dropout, initialize=True)
    model = nn.DataParallel(atloc)
    '''--------------------Optimization----------------------'''
    param_list = [{'params': model.module.parameters()}]
    param_list.append({'params': [train_criterion.module.sax, train_criterion.module.saq]})
    optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
    start = 0
    '''--------------------Loading_Weights----------------------'''
    if opt.weights != None:
        print('Loading the model {}'.format(opt.weights))
        opt.weights = os.path.join(opt.models_dir, opt.weights)
        weights_filename = osp.expanduser(opt.weights)
        if osp.isfile(weights_filename):
            checkpoint = torch.load(weights_filename)
            load_state_dict(model, checkpoint['model_state_dict'])
            load_state_dict(train_criterion, checkpoint['criterion_state_dict'])
            if opt.optimizer == 'unchange':
                print('both up date')
                optimizer.load_state_dict(checkpoint['optim_state_dict'])
                for p in optimizer.state.keys():
                    param_state = optimizer.state[p]
                    param_state["exp_avg"] = param_state["exp_avg"].cuda()  # Putting the memento into the device
                    param_state["exp_avg_sq"] = param_state["exp_avg_sq"].cuda()  # Putting the memento into the device
            elif opt.optimizer == 'restart':
                print('one update')
                param_list = [{'params': model.module.parameters()}]
                optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
            print('Loaded weights from {:s}'.format(weights_filename))
            start = int(opt.weights.split("/")[-1].split(".")[0].split("_")[1]) + 1
        else:
            print('Could not load weights from {:s}'.format(weights_filename))
            sys.exit(-1)
    '''--------------------Transformation_Image----------------------'''
    stats_file = osp.join(opt.data_dir, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    tforms = [transforms.Resize(opt.cropsize)]
    tforms.append(transforms.RandomCrop(opt.cropsize))
    if opt.color_jitter > 0:
        assert opt.color_jitter <= 1.0
        print('Using ColorJitter data augmentation')
        tforms.append(transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter, hue=0.5))
    else:
        print('Not Using ColorJitter')
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
    image_transform = transforms.Compose(tforms)
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
    tforms.append(transforms.RandomCrop(opt.cropsize))
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
    depth_transform = transforms.Compose(tforms)
    '''-------------------Transformation_Depth_Test----------------'''
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
    kwargs = dict(scene=opt.scene, data_path=opt.data_dir, image_transform=image_transform, depth_transform=depth_transform,target_transform=target_transform, seed=opt.seed)
    val_kwargs = dict(scene=opt.scene, data_path=opt.data_dir, image_transform=image_transform_val, depth_transform=depth_transform_val, target_transform=target_transform, seed=opt.seed)
    if opt.model == 'AtLoc':
        if opt.dataset == '7Scenes':
            train_set = SevenScenes(train=True, **kwargs)
            val_set = SevenScenes(train=False, **val_kwargs)
        elif opt.dataset == 'RobotCar':
            train_set = RobotCar(train=True, **kwargs)
            val_set = RobotCar(train=False, **val_kwargs)
        else:
            raise NotImplementedError
    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
    '''--------------------Model_To_Cuda----------------------'''
    model = model.to(device)
    train_criterion = train_criterion.to(device)
    val_criterion = val_criterion.to(device)
    '''--------------------Validation_Beginning----------------------'''
    total_steps = opt.steps
    writer = SummaryWriter(log_dir=opt.runs_dir)
    experiment_name = opt.exp_name
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
    meminfo_start = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo_start)
    for epoch in range(start, opt.epochs): 
        if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1) or epoch==start:
            val_loss = AverageMeter()
            model.eval()
            end = time.time()
            val_data_time = AverageMeter()
            val_batch_time = AverageMeter()
            val_batch_storage = AverageMeter()
            #----------------------------
            for batch_idx, (val_rgb_var, val_depth_var, val_target) in enumerate(val_loader):
                val_data_time.update(time.time() - end)
                val_rgb_var = Variable(val_rgb_var, requires_grad=False).to(device)
                val_depth_var = Variable(val_depth_var, requires_grad=False).to(device)
                val_target_var = Variable(val_target, requires_grad=False).to(device)
                with torch.set_grad_enabled(False):
                    if opt.mode == "concatenate":
                        val_output = model(image=val_rgb_var, depth=val_depth_var, mode='val')
                    elif opt.mode == "vmloc":
                        val_output, _, _, _ = model(image=val_rgb_var, depth=val_depth_var, K=opt.K, epoch=epoch, mode='val')
                        val_output = val_output.view(val_rgb_var.size(0),opt.K,-1)
                        val_output = torch.mean(val_output,dim=1)
                        val_output = val_output.view(1, val_rgb_var.size(0), -1)
                    val_loss_tmp = val_criterion(val_output, val_target_var)
                    val_loss_tmp = val_loss_tmp.item()
                val_loss.update(val_loss_tmp)
                val_batch_time.update(time.time() - end)
                val_batch_storage.update(pynvml.nvmlDeviceGetMemoryInfo(handle).used - meminfo_start.used)
                writer.add_scalar('val_err', val_loss_tmp, total_steps)
                end = time.time()
            print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))
            print('------------{:f}----------------{:f}--------------'.format(train_criterion.module.sax.item(),train_criterion.module.saq.item()))
            print('------------Time consuming: {:f}---Storaing: {:f}---------'.format(val_batch_time.avg, val_batch_storage.avg))
            print('---------------------{}---------------------'.format(epoch))
            if epoch % opt.save_freq == 0:
                filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
                checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.module.state_dict()}
                torch.save(checkpoint_dict, filename)
                print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))
            #------------Recording----------------
            written(opt,('Val {:s}: Epoch {:d}, val_loss {:f}\n'.format(experiment_name, epoch, val_loss.avg)))
            #----------------------------
        
        model.train()
        kl_loss_reco = AverageMeter()
        dis_loss_reco = AverageMeter()
        end = time.time()
        train_data_time = AverageMeter()
        train_batch_time = AverageMeter()
        train_batch_storage = AverageMeter()
        for batch_idx, (rgb_var, depth_var, target) in enumerate(train_loader):
            train_data_time.update(time.time() - end)
            rgb_var = Variable(rgb_var, requires_grad=True).to(device)
            depth_var = Variable(depth_var, requires_grad=True).to(device)
            target_var = Variable(target, requires_grad=False).to(device)
            with torch.set_grad_enabled(True):
                if opt.mode == "concatenate":
                    if epoch < opt.train_mask:
                        rgb_var, depth_var = regular_mask_training(rgb_var, depth_var, batch_idx)
                    output = model(rgb_var, depth_var, mode='train')
                    loss_tmp = train_criterion(output, target_var)
                    kld = torch.tensor(0)
                    loss = loss_tmp                   
                elif opt.mode == "vmloc":
                    if epoch < opt.train_mask:
                        rgb_var, depth_var = regular_mask_training(rgb_var, depth_var, batch_idx)              
                    loss, kld, loss_tmp = loss_function_mvae(model, rgb_var, depth_var, target_var, epoch, opt, train_criterion, device, Type1=opt.mode)
            kl_loss_reco.update(torch.mean(kld.float()).item())
            dis_loss_reco.update(torch.mean(loss_tmp.float()).item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_batch_time.update(time.time() - end)
            train_batch_storage.update(pynvml.nvmlDeviceGetMemoryInfo(handle).used - meminfo_start.used)
            writer.add_scalar('train_err', torch.mean(loss_tmp.float()).item(), total_steps)
            end = time.time()
        print('Training {:s}: Epoch {:d}, kl_loss {:f}, dis_loss{:f}'.format(experiment_name, epoch, kl_loss_reco.avg, dis_loss_reco.avg))
        print('------------{:f}----------------{:f}--------------'.format(train_criterion.module.sax.item(),train_criterion.module.saq.item()))
        print('------------Time consuming: {:f}---Storaging: {:f}---------'.format(train_batch_time.avg, train_batch_storage.avg))
        print('---------------------{}---------------------'.format(epoch))
        '''------------Recording----------------'''
        if epoch % opt.save_freq == 0:
                filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
                checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.module.state_dict()}
                torch.save(checkpoint_dict, filename)
                print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))
        written(opt,('Training {:s}: Epoch {:d}, kl_loss {:f}, dis_loss{:f} \n'.format(experiment_name, epoch, kl_loss_reco.avg, dis_loss_reco.avg)))
        written(opt,('------------{:f}----------------{:f}--------------'.format(train_criterion.module.sax.item(),train_criterion.module.saq.item())))
        written(opt,('---------------------{}---------------------\n'.format(epoch)))
    writer.close()

def loss_function_mvae(model, rgb_var, depth_var, target_var, epoch, opt, train_criterion, device, Type1):
    if Type1 == "vmloc":
        output, feature, _, kl = model(image=rgb_var, depth=depth_var, K=opt.K, epoch=epoch, mode='train')
    try:
        batch_size =  rgb_var.size(0)
    except:
        batch_size = depth_var.size(0)
    output = output.view(opt.K, batch_size, -1)
    feature = feature.view(opt.K, batch_size, -1)
    kld = (kl[0].view(opt.K, batch_size) - kl[1].view(opt.K, batch_size)) * opt.kl_para
    s = output.size()
    with torch.set_grad_enabled(True):
        output = output.view(s[1], s[0], *s[2:])
        loss_tmp = train_criterion(output, target_var).permute(1, 0)  # Size (K,B)
        if epoch >= opt.train_mean:
            loss = dreg(kld, loss_tmp, feature, opt)
        else:
            kld = torch.tensor(0)
            loss = torch.mean(loss_tmp)
    return loss, - kld, loss_tmp

def regular_mask_training(img, dep, check): 
    if check%5 == 1:
        img = img*torch.tensor(0).cuda().float()
    elif check%5 == 2:
        dep = dep*torch.tensor(0).cuda().float()
    return img, dep

if __name__ == "__main__":
    main()
                                                   