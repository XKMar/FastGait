# FastGait 

The *official* implementation for the [Dynamic Aggregated Network for Gait Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_Dynamic_Aggregated_Network_for_Gait_Recognition_CVPR_2023_paper.pdf) which is accepted by [CVPR-2023](https://cvpr.thecvf.com/Conferences/2023).

## What's New

#### [04 Mar 2023]
+ We support DistributedDataParallel DDP which enables data parallel training.

## Installation

```shell
git clone https://github.com/XKMar/FastGait.git
cd FastGait
```


## Example #1:
Training on CASIA-B dataset .
+ STEP 1: Prepare the dataset and copy the directory into the config file ``DATA_ROOT`` (configs\benchmarks\casia-b_mixnet_gmpa.yaml).
+ STEP 2: Modify the directory where the log and model files ``LOGS_ROOT`` (configs\benchmarks\casia-b_mixnet_gmpa.yaml).

### Train

```shell
bash tools dist_train.sh
```

### Test

```shell
bash tools dist_test.sh
```


## Citation
If you find this code useful for your research, please cite our paper
```
@inproceedings{ma2023dynamic,
  title={Dynamic aggregated network for gait recognition},
  author={Ma, Kang and Fu, Ying and Zheng, Dezhi and Cao, Chunshui and Hu, Xuecai and Huang, Yongzhen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22076--22085},
  year={2023}
}
```
