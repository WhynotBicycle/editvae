from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
# torch.multiprocessing.set_start_method('spawn', force=True)

class BenchmarkDatasetBase(data.Dataset):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None, device='cpu', labs=False, segs=False, cleaned=False, unsup_lab=False, cuted=False, down_sample=True, duplicated=False, holes=False):
        self.npoints = npoints
        self.root = root
        self.catfile = 'code/TreeGAN/dataloader/synsetoffset2category.txt'
        # print(catfile)
        self.cat = {}
        self.uniform = uniform
        self.classification = classification
        self.device = device
        self.labs = labs
        self.unsup_lab = unsup_lab
        self.down_sample = down_sample
        
        if segs:
            if cleaned:
                file = 'cleaned_legs'
            else:
                file = 'legs'
        else:
            if unsup_lab:
                file = 'points_unsup'
            elif cleaned:
                file = 'cleaned_points'
            else:
                file = 'points'
            
        lab_file = file+'_label'
        if cuted:
            file = 'cuted'
        elif duplicated:
            file = 'duplicated'
        elif holes:
            file = 'holes'
        
#         import json
#         with open(root+'train_test_split/shuffled_test_file_list.json') as f:
#             sl = json.load(f)
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
                
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], file)
            dir_seg = os.path.join(self.root, self.cat[item], lab_file)
            dir_sampling = os.path.join(self.root, self.cat[item], 'sampling')
            
            fns = sorted(os.listdir(dir_point))

            for fn in fns:
#                 if 'shape_data/'+self.cat[item]+'/'+fn.split('.')[0] in sl:
                    
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'),\
                                        os.path.join(dir_seg, token + '.seg'),\
                                        os.path.join(dir_sampling, token + '.sam'),\
                                        token))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        
    def __len__(self):
        return len(self.datapath)
    
class BenchmarkDataset(BenchmarkDatasetBase):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None, device='cpu', labs=False, segs=False, cleaned=False, unsup_lab = False, cuted=False, down_sample=True, duplicated=False, holes=False):
        super().__init__(root, npoints, uniform, classification, class_choice, device, labs, segs, cleaned, unsup_lab, cuted, down_sample, duplicated, holes)
        
        self.whole = []
        for fn in self.datapath:
            cls = self.classes[fn[0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64)
            if self.labs:
                lab = fn[4]
            
            if self.uniform:
                choice = np.loadtxt(fn[3]).astype(np.int64)
                assert len(choice) == self.npoints, "Need to match number of choice(2048) with number of vertices."
            else:
                choice = np.random.randint(0, len(seg), size=self.npoints)
                    
            if self.down_sample:   
                point_set = point_set[choice]
            seg = seg[choice]

            point_set = torch.from_numpy(point_set)
            seg = torch.from_numpy(seg)
            cls = torch.from_numpy(np.array([cls]).astype(np.int64))
            
            if self.labs:
                self.whole.append((point_set.to(self.device), seg.to(self.device), lab))
            else:
                if self.classification:
                    self.whole.append((point_set.to(self.device), cls.to(self.device)))
                else:
                    self.whole.append((point_set.to(self.device), seg.to(self.device)))
            # if self.labs:
            #     self.whole.append((point_set, seg, lab))
            # else:
            #     if self.classification:
            #         self.whole.append((point_set, cls))
            #     else:
            #         self.whole.append((point_set, seg))
                
    def __getitem__(self, index):
        return self.whole[index]
    
class BenchmarkDatasetOnTheFly(BenchmarkDatasetBase):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None, device='cpu', labs=False, segs=False, cleaned=False, unsup_lab = False, cuted=False, down_sample=True, duplicated=False, holes=False):
        super().__init__(root, npoints, uniform, classification, class_choice, device, labs, segs, cleaned, unsup_lab, cuted, down_sample, duplicated, holes)
        
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        if self.labs:
            lab = fn[4]
        
        if self.uniform:
            choice = np.loadtxt(fn[3]).astype(np.int64)
            assert len(choice) == self.npoints, "Need to match number of choice(2048) with number of vertices."
        else:
            if len(seg) < point_set.shape[0]:
                choice = np.random.randint(0, len(seg), size=self.npoints)
            else:
                choice = np.random.randint(0, point_set.shape[0], size=self.npoints)
#             choice = np.random.randint(0, len(seg), size=self.npoints)
                
        if self.down_sample:
            point_set = point_set[choice]
        seg = seg[choice]

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        if self.labs:
            return point_set.to(self.device), seg.to(self.device), lab
        else:
            if self.classification:
                return point_set.to(self.device), cls.to(self.device)
            else:
                return point_set.to(self.device), seg.to(self.device)
        # if self.labs:
        #     return point_set, seg, lab
        # else:
        #     if self.classification:
        #         return point_set, cls
        #     else:
        #         return point_set, seg

    
