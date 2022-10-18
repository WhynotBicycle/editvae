import torch
from loss_function import ChamferDistance, Calus, _remove_zero_rows
from torch.nn.utils.rnn import pad_sequence
import subprocess
import time
import numpy as np

import sys

from PyTorchEMD.emd import earth_mover_distance
from JSD import jsd_between_point_cloud_sets

def get_gpu_memory_map():   
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    
    return float(result)
## return the min distance between x and a list of y
## x is a 3D point cloud as torch.tensor(n_pts, 3), y_list is a list of point clouds
def minChamferDistance(x, y_list, device=None):
    cd_list = []
    for y in y_list:
        cd,_,_ = Calus(x, y, device=device)
        cd_list.append(cd.item())
    
    return min(cd_list)

## duplicated codes for saving computational time
def CDs(x, y_list):
    cd_list = []
    for y in y_list:
        cd,_,_ = Calus(x, y)
        cd_list.append(cd.item())
    
    return cd_list

def closestCD(x, y_list, device=None):
    cd_list = []
    for y in y_list:
        cd,_,_ = Calus(x, y, device=device)
        cd_list.append(cd.item())
    
    min_cd = min(cd_list)
    
    return cd_list.index(min_cd), min_cd 

def get_min_dist_list(x_list, y_list, device=None):
    dist_list = []
    for x in x_list:
        dist_list.append(minChamferDistance(x, y_list, device))
    return dist_list

## return the index in the current batch
def get_closests(x_list, y_list, device=None):
    closest_list = []
    dist_list = []
    for x in x_list:
        index, dist = closestCD(x, y_list, device=device)
        closest_list.append(index)
        dist_list.append(dist)
    return closest_list, dist_list

## the inpout is list of point clouds 1*n_pts*3
## for each groundtruth ptc, return the min EMD in generated list
def _emd(groundtruth_list, generated_list, device=None):
    emds = []
    min_index = []
    for gt in groundtruth_list:
        sub_emds = []
        
#         c_time = time.time()
        for generated in generated_list:
            d = earth_mover_distance(generated, gt, transpose=False)/generated.size(1)
#             d = earth_mover_distance(gt, generated, transpose=False)/gt.size(1)
            sub_emds.append(d)
#         print(time.time() - c_time)
        emds.append(min(sub_emds))
        min_index.append(sub_emds.index(min(sub_emds)))
    
    return min_index, emds
#     return emds, min_index

def mmd_from_generated(dataloader, generated_path, whole_batchs, n_train_batchs, distance='CD'):
    min_distance = []
    for _iter, data in enumerate(dataloader):
        ipts = data.cuda()
        ipts = list(torch.split(ipts, 1, dim=0))
        ipts = [pt for pt in ipts if pt.sum() != 0]
        
        min_distance_candidates = []
        for j in range(whole_batchs):
            if j >= n_train_batchs:
                generated = torch.load(generated_path+str(j)+'.pt').cuda()
                generated = list(torch.split(generated, 1, dim=0)) # a list of 1 * n_pts * 3

                ## for each generated pcls, calculate the shortest distance to the batch of original pcls
                if distance == 'CD':
                    batch_min_distance = get_min_dist_list(ipts, generated)
                elif distance == 'EMD':
                    _, batch_min_distance = _emd(ipts, generated)

                min_distance_candidates.append(batch_min_distance)
                
        ## for this batch of generated pcls, calculate the min distance in all original pcls
        sub_min_distance = [min(dist) for dist in zip(*min_distance_candidates)]
        min_distance = min_distance + sub_min_distance
        print('MMD up to now: '+ str(sum(min_distance)/float(len(min_distance))))

    return sum(min_distance)/float(len(min_distance))

def coverage_from_generated(dataloader, generated_path, whole_batchs, n_train_batchs, batch_size, distance='CD'):
    min_index = []
    total_pts = (whole_batchs - n_train_batchs) * batch_size

    for _iter in range(whole_batchs):
        if _iter >= n_train_batchs:
            generated = torch.load(generated_path+str(_iter)+'.pt').cuda()
            generated = list(torch.split(generated, 1, dim=0)) # a list of 1 * n_pts * 3
            
            closest_dists = []
            closest_index = []
            for j, data in enumerate(dataloader):
                ipts = data.cuda()
                ipts = list(torch.split(ipts, 1, dim=0))
                ipts = [pt for pt in ipts if pt.sum() != 0]
                
                if distance == 'CD':
                    closests, dist = get_closests(generated, ipts)
                elif distance == 'EMD':
                    closests, dist = _emd(generated, ipts)

                closests = [ (j+n_train_batchs) * batch_size + x for x in closests]
                closest_index.append(closests)
                closest_dists.append(dist)

            relative_index = [dist.index(min(dist)) for dist in zip(*closest_dists)]
            sub_min_index = []
            cl_i = list(map(list, zip(*closest_index)))
            for i in range(len(relative_index)):
                sub_min_index.append(cl_i[i][relative_index[i]])
            min_index = min_index + sub_min_index
            print('Coverage up to now: '+ str(len(set(min_index))/float(total_pts)) )
            
    return len(set(min_index))/float(total_pts)
    
def _parts_pt(point, seg, framework=None):
    cluster_num = torch.unique(seg).size(0)
    pt_size = point.size()
    seg_size = seg.size()
    assert pt_size[0] == seg_size[0]
    assert pt_size[1] == seg_size[1]
    
    if framework is not None:
        rg = range(1, 2)
    else:
        rg = range(1, cluster_num+1)

    pt_list = []
    for b in range(pt_size[0]):
        pt = point[b,:,:]
        sg = seg[b,:]

        for n in rg:
            idx = (sg==n).nonzero().squeeze(1)
            sub_pt = torch.index_select(pt, 0, idx)
            if sub_pt.size(0) == 0:
                continue
            pt_list.append(sub_pt)
            
            
    pt_list = pad_sequence(pt_list).transpose(0, 1)

    return pt_list  
    
def symmetry_pt(pt):
    idx = torch.randint(0, pt.size(1), (int(pt.size(1)/2), ))
    pt = pt[:, idx, :]

    pt2 = pt.clone()
    pt2[:,:,2] = -pt2[:,:,2]

    return torch.cat((pt, pt2), 1)
    
def mmd_from_generated_test(dataloader, generated_path, device, whole_batchs, n_train_batchs, train=True, distance='CD', parts_based=False, selected=False, framework=None, symmetry=False):
    ## to fix a bug on EMD immplmenation
    if distance == 'EMD':
        gpu = int(str(device)[-1])
        device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
    
    min_distance = []
    for _iter, data in enumerate(dataloader):
        ipts, seg = data
        if symmetry:
            ipts = symmetry_pt(ipts)
        ipts = ipts.to(device)
        
        if selected: ## remove cluster_num = 4
            new_ipts = []
            for b in range(ipts.size(0)):
                sg = seg[b,:]
                if torch.unique(sg).size(0) == 3 and torch.unique(sg)[2] != 4:
                    new_ipts.append(ipts[b,:,:].unsqueeze(0))
            ipts = torch.cat(new_ipts, dim=0)
#             print(ipts.size())
                
        if parts_based or framework is not None:
            seg = seg.to(device)
            ipts = _parts_pt(ipts, seg, framework=framework)
           
        ipts = list(torch.split(ipts, 1, dim=0))
        ipts = [pt for pt in ipts if pt.sum() != 0]
        
        min_distance_candidates = []
        for j in range(whole_batchs):
            if (train and j < n_train_batchs) or ((not train) and j >= n_train_batchs):
                generated = torch.load(generated_path+str(j)+'.pt')
                generated = generated.to(device)
                generated = list(torch.split(generated, 1, dim=0)) # a list of 1 * n_pts * 3

                ## for each generated pcls, calculate the shortest distance to the batch of original pcls
                if distance == 'CD':
                    batch_min_distance = get_min_dist_list(ipts, generated, device)
                elif distance == 'EMD':
                    _, batch_min_distance = _emd(ipts, generated, device)

                min_distance_candidates.append(batch_min_distance)
                
        ## for this batch of generated pcls, calculate the min distance in all original pcls
        sub_min_distance = [min(dist) for dist in zip(*min_distance_candidates)]
        min_distance = min_distance + sub_min_distance
        print('MMD up to now: '+ str(sum(min_distance)/float(len(min_distance))))

    return sum(min_distance)/float(len(min_distance))

def _2parts_list(point, seg, sq=False, offset=1, cluster_num=None):
    pt_size = point.size()
    seg_size = seg.size()
    assert pt_size[0] == seg_size[0]
    assert pt_size[1] == seg_size[1]

    pt_list = []
    for n in range(offset, cluster_num+offset):
        sub_pt_list = []
        for b in range(pt_size[0]):
            pt = point[b,:,:]
            sg = seg[b,:]
            
#             idx = (sg==n).nonzero().squeeze(1)
            idx = (sg==n).nonzero(as_tuple=True)[0]
            sub_pt = torch.index_select(pt, 0, idx)
            
            if sub_pt.size(0) == 0:
                continue
            if sq:
                sub_pt_list.append(sub_pt)
            else:
                sub_pt_list.append(sub_pt.unsqueeze(0))
        pt_list.append(sub_pt_list)
    return pt_list 

def pc_normalize(pc):
    centroid = pc.mean(dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.sum(pc**2, dim=1).sqrt().max()
    pc = pc / m
    return pc

def p2p_mmd_from_generated(dataloader, generated_path, device, whole_batchs, n_train_batchs, train=True, distance='CD', n_parts=4, spliter=None, offset=0, normalization=False):
    ## to fix a bug on EMD immplmenation
    if distance == 'EMD':
        gpu = int(str(device)[-1])
        device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
    
    r_parts = n_parts
    parts_means = 0
    ## for each parts
    for n in range(n_parts):
        min_distance = []
        for _iter, data in enumerate(dataloader):
            if spliter is None:
                ipts, seg = data
                ipts = ipts.to(device)
                seg = seg.to(device)
            else:
                ipts, _ = data
                ipts = ipts.to(device)
                seg = spliter(ipts)
                
#             m_value =  torch.sum(ipts**2, dim=1).sqrt().max()
                
            ipts = _2parts_list(ipts, seg, offset=offset, cluster_num=n_parts)[n]
            if normalization:
                ipts = [pc_normalize(p) for p in ipts]
    #         current_time = time.time()
            min_distance_candidates = []
            for j in range(whole_batchs):
                if ((train and j < n_train_batchs) or ((not train) and j >= n_train_batchs)) and (j-n_train_batchs) % n_parts == n:
                    generated = torch.load(generated_path+str(j)+'.pt')
                    generated = generated.to(device)
                    generated = list(torch.split(generated, 1, dim=0)) # a list of 1 * n_pts * 3

                    ## for each generated pcls, calculate the shortest distance to the batch of original pcls
                    if distance == 'CD':
                        batch_min_distance = get_min_dist_list(ipts, generated, device)
                    elif distance == 'EMD':
                        _, batch_min_distance = _emd(ipts, generated, device)

                    min_distance_candidates.append(batch_min_distance)

            ## for this batch of generated pcls, calculate the min distance in all original pcls
            sub_min_distance = [min(dist) for dist in zip(*min_distance_candidates)]
            min_distance = min_distance + sub_min_distance
            if len(min_distance) != 0:
                print('MMD for '+str(n)+'th part up to now: '+ str(sum(min_distance)/float(len(min_distance)) ))
            else:
                print('MMD for '+str(n)+'th part up to now: empty')
        if len(min_distance) != 0:
            parts_means += sum(min_distance)/float(len(min_distance))
        else:
            r_parts = r_parts - 1

    return parts_means/r_parts

def coverage_from_generated_test(dataloader, generated_path, device, whole_batchs, n_train_batchs, batch_size, train=True, distance='CD', framework=None, symmetry=False):
    min_index = []
    
    if train:
        total_pts = n_train_batchs * batch_size
    else:
        total_pts = (whole_batchs - n_train_batchs) * batch_size
    current_pts = lambda t: -1 if t else n_train_batchs-1
    
    if distance == 'EMD':
        gpu = int(str(device)[-1])
        device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
    
    for _iter in range(whole_batchs):
        if (train and _iter < n_train_batchs) or ((not train) and _iter >= n_train_batchs):
#             print('Processing batch No. ', _iter)
            generated = torch.load(generated_path+str(_iter)+'.pt')
            generated = generated.to(device)
            generated = list(torch.split(generated, 1, dim=0)) # a list of 1 * n_pts * 3
            
            closest_dists = []
            closest_index = []
            for j, data in enumerate(dataloader):
                ipts, seg = data
                if symmetry:
                    ipts = symmetry_pt(ipts)
                ipts = ipts.to(device)
                if framework is not None:
                    seg = seg.to(device)
                    ipts = _parts_pt(ipts, seg, framework=framework)
                ipts = list(torch.split(ipts, 1, dim=0))
                ipts = [pt for pt in ipts if pt.sum() != 0]
                
                if distance == 'CD':
#                     closests, dist = get_closests(ipts, generated, device)
                    closests, dist = get_closests(generated, ipts, device)
                elif distance == 'EMD':
                    closests, dist = _emd(generated, ipts, device)
#                     closests, dist = _emd(ipts, generated, device)

                # from relative index within batch to the absolute index
                if train:
                    closests = [ j * batch_size + x for x in closests]
                else:
                    closests = [ (j+n_train_batchs) * batch_size + x for x in closests]
                    
                closest_index.append(closests)
                closest_dists.append(dist)

            relative_index = [dist.index(min(dist)) for dist in zip(*closest_dists)]
            sub_min_index = []
            cl_i = list(map(list, zip(*closest_index)))
            for i in range(len(relative_index)):
                sub_min_index.append(cl_i[i][relative_index[i]])
            min_index = min_index + sub_min_index
            print('Coverage up to now: '+ str(len(set(min_index))/float(total_pts)) )
            
    return len(set(min_index))/float(total_pts)
            
    
def p2p_coverage_from_generated(dataloader, generated_path, device, whole_batchs, n_train_batchs, batch_size, train=True, distance='CD', n_parts=4, spliter=None, offset=0):
    ## to fix a bug on EMD immplmenation
    if train:
        total_pts = n_train_batchs * batch_size
    else:
        total_pts = (whole_batchs - n_train_batchs) * batch_size
    current_pts = lambda t: -1 if t else n_train_batchs-1
    
    if distance == 'EMD':
        gpu = int(str(device)[-1])
        device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
    
    parts_means = 0
    ## for each parts
    for n in range(n_parts):
        min_cov = []
        for _iter in range(whole_batchs):
            if ((train and _iter < n_train_batchs) or ((not train) and _iter >= n_train_batchs)) and (_iter-n_train_batchs) % n_parts == n:
                generated = torch.load(generated_path+str(_iter)+'.pt')
                generated = generated.to(device)
                generated = list(torch.split(generated, 1, dim=0)) # a list of 1 * n_pts * 3

                closest_dists = []
                closest_index = []
                for j, data in enumerate(dataloader):
                    if spliter is None:
                        ipts, seg = data
                        ipts = ipts.to(device)
                        seg = seg.to(device)
                    else:
                        ipts, _ = data
                        ipts = ipts.to(device)
                        seg = spliter(ipts)
                    ipts = _2parts_list(ipts, seg, offset=offset, cluster_num=n_parts)[n]

                    if distance == 'CD':
                        closests, dist = get_closests(ipts, generated)
                    elif distance == 'EMD':
                        closests, dist = _emd(ipts, generated, device)

                    # from relative index within batch to the absolute index
                    if train:
                        closests = [ j * batch_size + x for x in closests]
                    else:
                        closests = [ (j+n_train_batchs) * batch_size + x for x in closests]

                    closest_index.append(closests)
                    closest_dists.append(dist)
                    
                relative_index = [dist.index(min(dist)) for dist in zip(*closest_dists)]
                sub_min_cov = []
                cl_i = list(map(list, zip(*closest_index)))
                for i in range(len(relative_index)):
                    sub_min_cov.append(cl_i[i][relative_index[i]])
                min_cov = min_cov + sub_min_cov
                print('Coverage for '+str(n)+'th part up to now: '+ str(len(set(min_cov))/float(total_pts)) )
        parts_means += len(set(min_cov))/float(total_pts)

    return parts_means/n_parts

def JSD_generated_test(dataloader, generated_path, whole_batchs, n_train_batchs, train=False, framework=None, symmetry=False):
    
    n = 0
    ## load the generated data
    for _iter in range(whole_batchs):
        if (train and _iter < n_train_batchs) or ((not train) and _iter >= n_train_batchs):
            gt = torch.load(generated_path+str(_iter)+'.pt')
            
            if framework is not None:
                gt = torch.cat(list(torch.split(gt, 1, dim=0)), dim=1)
                
            gt = gt.cpu().detach().numpy()
            
            if n is 0:
                gts = gt
            else:
                gts = np.concatenate((gts, gt), axis=1)
            n += 1
    
    gts = np.array(gts)/(2*np.sqrt(3))
    
    n = 0
    for _iter, data in enumerate(dataloader):
        ipt, seg = data
        if symmetry:
            ipt = symmetry_pt(ipt)
        if framework is not None:
            ipt = _parts_pt(ipt, seg, framework=framework)
            
            for b in range(ipt.size(0)):
                sub_ipt = ipt[b]
                idx = sub_ipt.sum(1).nonzero(as_tuple=True)[0]
                sub_ipt = torch.index_select(sub_ipt, 0, idx).unsqueeze(0)
                sub_ipt = sub_ipt.detach().cpu().numpy()
                
                if n is 0 and b is 0:
                    ipts = sub_ipt
                else:
                    ipts = np.concatenate((ipts, sub_ipt), axis=1)
                n += 1
        else:
            ipt = ipt.detach().cpu().numpy()
            if n is 0:
                ipts = ipt
            else:
                ipts = np.concatenate((ipts, ipt), axis=1)
            n += 1
              
    ipts = np.array(ipts)/(2*np.sqrt(3))
    
    return jsd_between_point_cloud_sets(gts, ipts)


def p2p_jsd_from_generated(dataloader, generated_path, whole_batchs, n_train_batchs, train=True, n_parts=4, spliter=None, device=None, offset=0):
    parts_means = 0
    ## for each parts
    for num in range(n_parts):
#         if num > 2:
#             break
        n = 0
        ## load the generated data
        for j in range(whole_batchs):
            if ((train and j < n_train_batchs) or ((not train) and j >= n_train_batchs)) and (j-n_train_batchs) % n_parts == num:
                gt = torch.load(generated_path+str(j)+'.pt')
                gt = torch.cat(list(torch.split(gt, 1, dim=0)), dim=1)
                gt = gt.cpu().detach().numpy()
                
                if n is 0:
                    gts = gt
                else:
                    gts = np.concatenate((gts, gt), axis=1)
                n += 1

        gts = np.array(gts)/(2*np.sqrt(3))

        n = 0
        for _iter, data in enumerate(dataloader):
            if spliter is None:
                ipt, seg = data
                ipt = ipt.to(device)
                seg = seg.to(device)
            else:
                ipt, _ = data
                ipt = ipt.to(device)
                seg = spliter(ipt)
            ipt = _2parts_list(ipt, seg, offset=offset, cluster_num=n_parts)
            
#             print(len(ipt[num]))
            if len(ipt[num]) == 0:
                ipt = torch.ones(1, 1, 3).to(device)
            else:
                ipt = torch.cat(ipt[num], dim=1)
#             print(ipt.size())
            ipt = ipt.detach().cpu().numpy()
            
            
            if n is 0:
                ipts = ipt
            else:
                ipts = np.concatenate((ipts, ipt), axis=1)
            n += 1

        ipts = np.array(ipts)/(2*np.sqrt(3))
        tmp = jsd_between_point_cloud_sets(gts, ipts)
        print(tmp)
        parts_means += jsd_between_point_cloud_sets(gts, ipts)
        

    return parts_means/n_parts



def JSD_generated(dataloader, generated_path, whole_batchs, n_train_batchs):
    n = 0
    ## load the generated data
    for _iter in range(whole_batchs):
        if _iter >= n_train_batchs:
            gt = torch.load(generated_path+str(_iter)+'.pt').cuda()
            gt = gt.cpu().detach().numpy()
            
            if n is 0:
                gts = gt
            else:
                gts = np.concatenate((gts, gt), axis=1)
            n += 1
    gts = np.array(gts)/(2*np.sqrt(3))
    
    n = 0
    for _iter, data in enumerate(dataloader):
        ipt = data.cuda()
        ipt = ipt.detach().cpu().numpy()
        if n is 0:
            ipts = ipt
        else:
            ipts = np.concatenate((ipts, ipt), axis=1)
        n += 1
    ipts = np.array(ipts)/(2*np.sqrt(3))
    
    return jsd_between_point_cloud_sets(gts, ipts)