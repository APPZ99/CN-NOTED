import os
from tqdm import tqdm
import torch
import open3d as o3d
import numpy as np
import pickle

from model import nerf, camera
from tasks import lidar
from utils import config, misc

def main():
    # 加载参数
    args = config.load_parser()
    
    # 判定运行模式：debug 和 finetune（微调）模式
    if args.use_debug_mode:
        torch.autograd.set_detect_anomaly(True)
    
    if args.finetune:
        coarse_poses = misc.load_final_poses(args)
    
    # 获取序列中的关键帧
    key_frame_info = misc.load_keyframe_info(args)
    
    # 存放关键帧和初始位姿
    key_frames = []
    init_pose = []
    
    # 点云数据路径
    pcd_path = os.path.join(args.scene_path, "pointcloud")
    # 遍历点云数据
    for pcd_num, pcd_file in enumerate(tqdm(sorted(os.listdir(pcd_path), key=lambda s: int(s.split('.')[0])), 
                                            desc = "key frames")):
        # 读取点云数据
        pcd_file = os.path.join(pcd_path, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd = np.asarray(pcd.points)
        # 部署到 GPU
        pcd = torch.from_numpy(pcd).float().to(args.device)

        # 把点云数据添加到关键帧列表中
        key_frames.append(pcd)
        
        # 根据是否微调添加不同的初始位姿信息
        if args.finetune:
            init_pose.append(coarse_poses[pcd_num])
        else:
            init_pose.append(key_frame_info[pcd_num][1])
            
    # 初始化位姿
    init_pose = torch.stack(init_pose, 0)
    
    # 创建 NeRF 模型：输入网络深度、网络宽度、网格大小
    nerf_model = nerf.occNeRF(args.netdepth, args.netwidth, bbox = args.bbox).to(args.device)
    # 创建相机位姿学习模型
    pose_model = camera.LearnPose(len(key_frames), init_pose).to(args.device)
    # 添加观测点路径
    ckpt_path = os.path.join(args.scene_path, "checkpoints")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    poses = lidar.global_op(key_frames, nerf_model, pose_model, args)
    
    save_pickle = os.path.join(args.scene_path, "final_pose.pkl")
    with open(save_pickle, "wb") as pickle_file:
        pickle.dump(poses, pickle_file)
    
    save_model = os.path.join(args.scene_path, "final_map.pt")
    torch.save(nerf_model.state_dict(), save_model)

if __name__ == "__main__":
    main()
    