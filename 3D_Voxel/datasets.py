import os
import json
import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def normalize_point_cloud(point_cloud):
    """
    Normalize the point cloud (only the 3D coordinates) to fit within a unit cube [0, 1]^3.

    Parameters:
    - point_cloud: Nx6 array of points (x, y, z, r, g, b).

    Returns:
    - normalized_point_cloud: Normalized Nx3 point cloud (only x, y, z).
    """
    # Extract only the (x, y, z) coordinates
    xyz_coords = point_cloud[:, :3]

    # Find the minimum and maximum values for normalization
    min_bound = np.min(xyz_coords, axis=0)
    max_bound = np.max(xyz_coords, axis=0)

    # Normalize to the range [0, 1]
    normalized_xyz = (xyz_coords - min_bound) / (max_bound - min_bound)

    return normalized_xyz


class ModelNetDataset(Dataset):
    def __init__(self, root, num_category, split='train', resolution=32, process_data=False):
        self.root = root
        self.npoints = -1 # -1 means no sampling #args.num_point=1024
        self.process_data = process_data
        self.resolution = resolution
        self.uniform = False #args.use_uniform_sample
        self.use_normals = True #args.use_normals
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        voxel_grid,label=self.voxelize(point_set,label[0])
        return torch.tensor(voxel_grid).float(), label

    def __getitem__(self, index):
        return self._get_item(index)

    def voxelize(self, point, label, voxel_dim=32):
        point_cloud_xyz = normalize_point_cloud(point)
        voxel_grid = np.zeros((1,voxel_dim, voxel_dim, voxel_dim), dtype=np.uint8)
        voxel_coords = (point_cloud_xyz * (voxel_dim - 1)).astype(int)
        voxel_grid[0,voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
        return voxel_grid, label

class ShapeNetDataset(Dataset):
    def __init__(self,root, npoints=-1, split='train', resolution=32, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        # self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
        #                     'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
        #                     'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
        #                     'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
        #                     'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    def __getitem__(self, index):
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        voxel_grid, cls = self.voxelize(point_set, cls[0])
        # point_set = data[:, 0:3]
        # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return  torch.tensor(voxel_grid).float(), cls

    def __len__(self):
        return len(self.datapath)

    def voxelize(self, point, label, voxel_dim=32):
        point_cloud_xyz = normalize_point_cloud(point)
        voxel_grid = np.zeros((1, voxel_dim, voxel_dim, voxel_dim), dtype=np.uint8)
        voxel_coords = (point_cloud_xyz * (voxel_dim - 1)).astype(int)
        voxel_grid[0, voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
        return voxel_grid, label