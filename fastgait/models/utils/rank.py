import csv
import json
import numpy as np

import torch
from .build_dist import build_dist


def de_diag(acc, each_angle=False):
    div_num = acc.shape[1] -1
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / div_num
    if not each_angle:
        result = np.mean(result)
        
    return result

def eval_datasets(cfg, features, label, views, types, dataset_name, max_rank):
    """evaluate the casia-b, casia-c, oumvlp datasets
    Args:
        features: torch.list. The features of test sets.
        label: list. The label list of test sets
        views: list. The view list of test sets
        types: list. The type list of test sets
        dataset_name: str. The name of the datasets
        max_rank: int. Number of the rank.  
    """

    view_set = sorted(list(set(views)))

    probe_seq_dict  =  {'CASIA-B': {'types':[['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']], 'views': [[v] for v in view_set]},
                        'OUMVLP':  {'types':[['00']], 'views': [[v] for v in view_set]},
                        'CASIA-C': {'types':[['H_scene1_nm_1', 'H_scene1_nm_2', 'H_scene2_nm_1', 'H_scene2_nm_2', 'L_scene1_nm_1', 'L_scene1_nm_2', 'L_scene2_nm_1', 'L_scene2_nm_2'], 
                                        ['H_scene1_bg_1', 'H_scene1_bg_2', 'H_scene2_bg_1', 'H_scene2_bg_2', 'L_scene1_bg_1', 'L_scene1_bg_2', 'L_scene2_bg_1', 'L_scene2_bg_2'],
                                        ['H_scene1_cl_1', 'H_scene1_cl_2', 'H_scene2_cl_1', 'H_scene2_cl_2', 'L_scene1_cl_1', 'L_scene1_cl_2', 'L_scene2_cl_1', 'L_scene2_cl_2']],
                                    'views': [[v] for v in view_set]},
                        'FSCL':    {'types':[['type1'],['type2'],['type3'],['type4']], 'views': None},}

    gallery_seq_dict = {'CASIA-B': {'types':[['nm-01', 'nm-02', 'nm-03', 'nm-04']], 'views': [[v] for v in view_set]},
                        'OUMVLP':  {'types':[['01']], 'views': [[v] for v in view_set]},
                        'CASIA-C': {'types':[['H_scene3_nm_1', 'H_scene3_nm_2', 'L_scene3_nm_1', 'L_scene3_nm_2']], 'views': [[v] for v in view_set]},
                        'FSCL':    {'types':[['type1']], 'views':[['camera1_view0_0'], ['camera1_view1_0'], ['camera1_view2_0']]},}

    #TODO: calculate the CMC, mAP, etc

    type_num = len(probe_seq_dict[dataset_name]['types'])
    if probe_seq_dict[dataset_name]['views'] is not None:
        probe_num = len(probe_seq_dict[dataset_name]['views'])
    else:
        probe_num = 1
    gallery_num = len(gallery_seq_dict[dataset_name]['views'])

    CMC  = np.zeros([type_num, probe_num, gallery_num, max_rank])

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset_name]['types']):
        for gallery_seq in gallery_seq_dict[dataset_name]['types']:
            for v1 in range(probe_num): # probe view
                for v2 in range(gallery_num): # gallery view
                    # acquire the gallery seqs 
                    gseq_mask = np.isin(types, gallery_seq) & np.isin(views, gallery_seq_dict[dataset_name]['views'][v2])
                    gallery_x = features[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    # acquire the probe seqs
                    if probe_seq_dict[dataset_name]['views'] is not None:
                        pseq_mask = np.isin(types, probe_seq) & np.isin(views, probe_seq_dict[dataset_name]['views'][v1])
                    else:
                        # the num of views is unfixed in some real datasets
                        if probe_seq == gallery_seq:
                            pseq_mask = np.isin(types, probe_seq) & np.isin(views, gallery_seq_dict[dataset_name]['views'][v2], invert=True)
                        else:
                            pseq_mask = np.isin(types, probe_seq)
                    
                    probe_x = features[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    # calculate the distance between probe and gallery 
                    dist = build_dist(cfg, probe_x, gallery_x) # numpy.ndarray
                    idx = np.argsort(dist, axis=1)

                    CMC[p, v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:max_rank]], 1) > 0,
                                0) * 100 / dist.shape[0], 2)

    return CMC

def evaluate_rank(
    cfg,
    features,
    label,
    views,
    types,
    dataset_name,
    max_rank=5,
    verbose=True,
):
    """Evaluates CMC rank and mAP.

    Args:
        features (tensor):  all features of test dataset
        label (numpy.ndarray): 1-D array all label of test dataset
        views (numpy.ndarray): 1-D array all views of test dataset
        types (numpy.ndarray): 1-D array all types of test dataset
        dataset_name (str): the name of dataset 
    """

    if dataset_name == "GREW":
        CMC = None
        rank = 20

        # 1. read submission.csv
        with open("records/results/submission.csv", 'r') as f:
            reader = csv.reader(f)
            listcsv = []
            for i, row in enumerate(reader):
                if i == 0:
                    print(row)
                listcsv.append(row)
        print('finish reading csv!')

        # 2. prepare probe & gallery
        # work_path = '/mnt/cfs/algorithm/users/xianda.guo/work/'
        # listCollection_probe_path = work_path + 'partition/all_list_grew_test_probe_iccv2021.npy
        pseq_mask = np.isin(label, 0)
        gseq_mask = np.isin(label, 0, invert=True)
        seq_type_list = types[pseq_mask]

        probe_x = features[pseq_mask, :]

        gallery_x = features[gseq_mask, :]
        gallery_y = label[gseq_mask]

        dist = build_dist(cfg, probe_x, gallery_x) 
        idx = np.argsort(dist, axis=1)

        for i, vidId in enumerate(seq_type_list):
            for j, _idx in enumerate(idx[i][:rank]):
                listcsv[i+1][0] = vidId
                listcsv[i+1][j + 1] = int(gallery_y[_idx])

        with open("records/results/submission_grew.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in listcsv:
                    writer.writerow(row)
        print('=== Finish GREW test ===')
    
    else:

        CMC = eval_datasets(cfg, features, label, views, types, dataset_name, max_rank)

        if verbose:
            
            # print the CMC rank
            sep = "*******************************"
            if dataset_name is not None:
                print(f"\n{sep} The accuracy of {dataset_name} {sep}\n")
                
            if dataset_name == 'CASIA-B' or dataset_name == 'CASIA-C':
                # Print rank-1 accuracy of the best model
                for i in range(1):
                    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
                    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                        np.mean(CMC[0, :, :, i]),
                        np.mean(CMC[1, :, :, i]),
                        np.mean(CMC[2, :, :, i])))

                # Print rank-1 accuracy of the best model，excluding identical-view cases
                for i in range(1):
                    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                        de_diag(CMC[0, :, :, i]),
                        de_diag(CMC[1, :, :, i]),
                        de_diag(CMC[2, :, :, i])))

                # Print rank-1 accuracy of the best model (Each Angle)
                np.set_printoptions(precision=2, floatmode='fixed')
                for i in range(1):
                    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
                    print('NM:', de_diag(CMC[0, :, :, i], True))
                    print('BG:', de_diag(CMC[1, :, :, i], True))
                    print('CL:', de_diag(CMC[2, :, :, i], True))

            elif dataset_name == 'OUMVLP':
                # Print rank-1 accuracy of the best model
                for i in range(1):
                    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
                    print('NM: %.3f' % (np.mean(CMC[0, :, :, i])))

                # Print rank-1 accuracy of the best model，excluding identical-view cases
                for i in range(1):
                    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                    print('NM: %.3f' % (de_diag(CMC[0, :, :, i])))

                # Print rank-1 accuracy of the best model (Each Angle)
                np.set_printoptions(precision=2, floatmode='fixed')
                for i in range(1):
                    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
                    print('NM:', de_diag(CMC[0, :, :, i], True))

            elif dataset_name == 'FSCL':
                # Print rank-1 accuracy of the best model
                for i in range(3):
                    print('===Views-%d ===' % (i + 1))
                    print('TP01: %.3f,\tTP02: %.3f,\tTP03: %.3f,\tTP04: %.3f' % (
                        np.mean(CMC[0, :, i, 0]),
                        np.mean(CMC[1, :, i, 0]),
                        np.mean(CMC[2, :, i, 0]),
                        np.mean(CMC[3, :, i, 0])))

            else:
                print('No such %s datasets' % dataset_name)

    del CMC
    torch.cuda.empty_cache()
