import os
LOCAL_DIRECTORY = "/mnt/nas/kaichen/eng/LOC/vmloc"
# Change to your directory
import torch
import numpy as np
import pickle
import os.path as osp
import sys
sys.path.append(LOCAL_DIRECTORY + "/robotcar/python/")
sys.path.append(LOCAL_DIRECTORY + "/robotcar/models/")
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from image2 import load_image as load_image_sdk
from camera_model import CameraModel
from tools.utils import process_poses, calc_vos_simple, load_image
from torch.utils import data
from functools import partial
from operator import itemgetter
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
import random

class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train,  image_transform=None, depth_transform=None, target_transform=None, seed=7, real=False, vo_lib='orbslam'):
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        np.random.seed(seed)
        # directories
        data_dir = osp.join(data_path,  scene)
        # decide which sequences to use
        if train:
            split_file = osp.join(data_dir, 'train_split.txt')
        else:
            split_file = osp.join(data_dir, 'test_split.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
            #seqs = [int(l.split('seq-')[-1]) for l in f if not l.startswith('#')]
        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int64)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            #seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            #p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') > 0 and n.find('._') < 0]
            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib), 'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)
                frame_idx = pss[:, 0].astype(np.int)
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int64)
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:12] for i in frame_idx]
                ps[seq] = np.asarray(pss)
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
        c_img = None
        d_img = None
        while (c_img is None) or (d_img is None):
            c_img = load_image(self.c_imgs[index])
            d_img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
        #img = [c_img, d_img]
        index -= 1
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        if self.image_transform is not None:
            img = self.image_transform(c_img)
        if self.depth_transform is not None:
            dep = self.depth_transform(d_img)
        return img, dep, pose

    def __len__(self):
        return self.poses.shape[0]


class RobotCar(data.Dataset):
    def __init__(self, scene, data_path, train, image_transform=None, depth_transform=None, target_transform=None, real=False, seed=None, undistort=False, vo_lib='stereo'):
        np.random.seed(seed)
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.undistort = undistort
        # directories
        #data_dir = osp.join(data_path, 'RobotCar', scene)
        data_dir = osp.join(data_path, scene)
        # decide which sequences to use
        if train:
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'test_split.txt')
        print(split_filename)
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]
        ps = {}
        ts = {}
        vo_stats = {}
        print(seqs)
        self.imgs = []
        self.deps = []
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)
            # read the image timestamps
            ts_filename = osp.join(seq_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]
            if real:  # poses from integration of VOs
                if vo_lib == 'stereo':
                    vo_filename = osp.join(seq_dir, 'vo', 'vo.csv')
                    p = np.asarray(interpolate_vo_poses(vo_filename, ts[seq], ts[seq][0]))
                elif vo_lib == 'gps':
                    vo_filename = osp.join(seq_dir, 'gps', 'gps_ins.csv')
                    p = np.asarray(interpolate_ins_poses(vo_filename, ts[seq], ts[seq][0]))
                else:
                    raise NotImplementedError
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'r') as f:
                    vo_stats[seq] = pickle.load(f)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
            else:  # GT poses
                pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            #self.imgs.extend([osp.join(seq_dir, 'stereo', 'centre', '{:d}.png'.format(t)) for t in ts[seq]])
            self.imgs.extend([osp.join(seq_dir, 'stereo', 'centre_processed', '{:d}.png'.format(t)) for t in ts[seq]])
            self.deps.extend([osp.join(seq_dir, 'depth_processed', '{:d}.png'.format(t)) for t in ts[seq]])
            #self.imgs.extend([osp.join(seq_dir, 'depth', '{:d}.png'.format(t)) for t in ts[seq]])
        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        for seq in seqs:
            if len(seqs) == 1:
                index_exist=np.where(~np.all(np.isnan(ps[seq]), axis=1) == True)
                ps[seq] = ps[seq][index_exist]
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                              align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                              align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
        self.gt_idx = np.asarray(range(len(self.poses)))
        if len(seqs) == 1:
            self.imgs = list(itemgetter(*index_exist[0].tolist())(self.imgs))
            self.deps = list(itemgetter(*index_exist[0].tolist())(self.deps))
        camera_model = CameraModel(LOCAL_DIRECTORY + "/robotcar/models", osp.join('stereo', 'centre'))
        self.im_loader = partial(load_image_sdk, model=camera_model)
    def __getitem__(self, index):
        img = None
        while img is None:
            if self.undistort:
                img = np.uint8(load_image(self.imgs[index], loader=self.im_loader))
                dep = np.uint8(load_image(self.deps[index]))
            else:
                img = load_image(self.imgs[index])
                dep = load_image(self.deps[index])
            pose = np.float32(self.poses[index])
            index += 1
        index -= 1
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.depth_transform is not None:
            dep = self.depth_transform(dep)
        return img, dep, pose
    def __len__(self):
        return len(self.poses)
    