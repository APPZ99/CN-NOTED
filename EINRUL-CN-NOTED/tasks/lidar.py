import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

import sys
sys.path.append("..")

from utils import render, misc


import time



def global_op(key_frames, nerf_model, pose_model, args):
    
    # 使用 OpenGL、pangolin作为可视化工具
    if args.use_visualization:
        import OpenGL.GL as gl
        import pangolin
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pangolin.ModelViewLookAt(2, -2, -2, 0, 0, 0, pangolin.AxisDirection.AxisNegY))
        handler = pangolin.Handler3D(scam)

        # Create Interactive View in window
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        dcam.SetHandler(handler)
    
    # 定义 nerf、pose 模型的优化器
    nerf_optim = optim.Adam(nerf_model.parameters(), lr=args.map_lr)
    pose_optim = optim.Adam(pose_model.parameters(), lr=args.pose_lr)
    
    # 创建学习率调度器实现多步学习率调度策略
    milestone = [args.pose_milestone]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(pose_optim, milestone)
    
    # 判断是否微调
    if args.finetune:
        iteration = args.iteration
    else:
        iteration = args.pose_milestone

    # while not pangolin.ShouldQuit():
    for iter_num in tqdm(range(iteration), 
                         desc = "iteration"):
        if args.use_visualization:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)
        
        # 保存射线和深度信息
        full_rays = []
        full_depth = []

        for pose_num, key_frame in enumerate(key_frames):
            pose = pose_model(pose_num)
            # 随机采样射线
            pcd = render.random_sample_pcd(key_frame, args.sample_rays)
            # 通过点云数据得到射线和深度信息
            random_rays, random_depth = render.pcd2ray(pcd)
            # 得到相机位姿下的射线
            random_rays = render.transform_ray(random_rays, pose)
            # 添加到各自的数据列表中
            full_rays.append(random_rays)
            full_depth.append(random_depth)
        # 对所有数据进行拼接
        full_rays = torch.cat(full_rays, 0)
        full_depth = torch.cat(full_depth, 0)
        
        # 通过渲染得到网格的占据值、占据标签、占据权重值、深度值
        occ, label, weight, depths = render.render_rays(full_rays, full_depth, nerf_model, 
                                                        args.ray_points, args.variance, args.weight_coe, 
                                                        stratified = args.stratified_portion)
        
        # 深度的 MSE 误差
        depth_loss = F.mse_loss(full_depth.squeeze(), depths)
        # 对预测和真实标签之间的二元交叉熵损失
        direct_loss = F.binary_cross_entropy(occ, label, weight)
        
        if args.finetune:
            # 法向量损失
            gradient, gradient_neighbor = render.render_normal(full_rays, full_depth, nerf_model, 
                                                               args.normal_eps)
            cos_sim = F.cosine_similarity(gradient, gradient_neighbor, -1)
            normal_loss = F.l1_loss(torch.ones_like(cos_sim), cos_sim)
        else:
            normal_loss = torch.tensor([0], device = args.device)
        
        loss = args.direct_loss_lambda * direct_loss + \
               args.depth_loss_lambda * depth_loss + \
               args.normal_loss_lambda * normal_loss
        
        # 清空梯度
        nerf_optim.zero_grad()
        pose_optim.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度优化
        nerf_optim.step()
        pose_optim.step()
        
        scheduler.step()
        
        if iter_num % args.log_step == args.log_step - 1:
            print("depth loss :", depth_loss.item())
            print("normal loss :", normal_loss.item())
            print("direct loss :", direct_loss.item())
        
        if iter_num % (args.log_step * 6) == (args.log_step * 6) - 1:
            save_model = os.path.join(args.scene_path, "checkpoints", str(iter_num) + "_map.pt")
            torch.save(nerf_model.state_dict(), save_model)
            
        if args.use_visualization:
            with torch.no_grad():
                for pose_num in range(len(key_frames)):
                    pose = pose_model(pose_num)
                    pose_np = pose.cpu().numpy()
                    
                    # Draw camera
                    gl.glLineWidth(1)
                    color = misc.rainbow(pose_num/len(key_frames), True)
                    gl.glColor3f(*color)
                    pangolin.DrawCamera(pose_np, 0.2, 0.75, 0.8)

            pangolin.glDrawColouredCube(0.2)
            pangolin.FinishFrame()
    
    poses = []
    with torch.no_grad():
        for pose_num in range(len(key_frames)):
            pose = pose_model(pose_num)
            poses.append(pose)
            
    return poses
