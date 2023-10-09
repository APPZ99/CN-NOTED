import torch

def gen_weight_label(dist, weight_coe):
    # 得到距离正负大小，大于0为1，小于0为-1，等于0为0
    sign = torch.sign(dist)
    # 得到相应的标签和反标签
    label = (sign + 1.0) / 2.0
    inv_label = (-sign + 1.0) / 2.0
    # 获得权重值
    # TODO: 为什么这么计算
    weight = torch.exp(-weight_coe * torch.mul(dist, dist)) * label + torch.ones_like(dist) * inv_label
    
    # using soft label
    # label = torch.sigmoid(label_coe * dist)
    
    return weight, label

def image2ray(H, W, intrinsic):
    device = intrinsic.device
    uu = torch.linspace(0, W-1, W, device = device)
    vv = torch.linspace(0, H-1, H, device = device)
    point_u, point_v = torch.meshgrid(uu, vv, indexing = "xy")
    
    point_u = point_u.contiguous().view(-1)
    point_v = point_v.contiguous().view(-1)

    point_x = (point_u - intrinsic[0][2]) / intrinsic[0][0]
    point_y = (point_v - intrinsic[1][2]) / intrinsic[1][1]
    point_z = torch.ones_like(point_x)
    
    rays_o = torch.zeros((point_x.shape[0], 3), device = device)
    rays_d = torch.stack([point_x, point_y, point_z], -1)
    depth = point_z.unsqueeze(-1)
    
    rays = torch.cat([rays_o, rays_d, depth], -1)
    
    return rays
    
def random_sample_pcd(pcd, sample_rays):
    sampler = torch.randint(pcd.shape[0], (sample_rays,))
    pcd = pcd[sampler]
    
    return pcd

def pcd2ray(pcd):
    # 雷达坐标系转换到相机坐标系
    # X -> Z   Y -> -X   Z -> -Y
    Tl2c = torch.tensor([[0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0]], device = pcd.device)
    # 得到相机坐标系下的点云数据
    # pcd.permute() 对点云数据进行转置
    pcd = Tl2c @ pcd.permute(1, 0)
    # 再次转置恢复原来格式
    pcd = pcd.permute(1, 0)
    # 取 Z 轴，即深度信息，并进行扩维（N,1）
    depth = pcd[:, 2].unsqueeze(-1)
    # 得到深度方向的方向向量
    rays_d = pcd / depth
    # 射线原点
    rays_o = torch.zeros_like(rays_d)
    # 组成射线
    rays = torch.cat([rays_o, rays_d], -1)
    
    return rays, depth

def transform_ray(rays, pose):
    # avoid inplace operation
    # 得到相机位姿下的射线原点
    rays_o = rays[:, :3] + pose[:3, 3]
    # 得到相机位姿下的射线方向
    rays_d = pose[:3, :3] @ rays[:, 3:6].transpose(1, 0)
    rays_d = rays_d.transpose(1, 0)
    # 得到相机位姿下从原点出发的射线
    rays = torch.cat([rays_o, rays_d], -1)
    
    return rays

def render_rays(rays, gt_depths, nerf_model, ray_points, variance, weight_coe, stratified = None):
    # 赋值数据
    sample_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    device = rays.device
    
    # sample and get z_val using gt_depth
    # 将观测值作为 GT
    # 通过 torch.clamp_min 确保最小距离不小于0
    # 最后对每一条射线生成一个服从标准正态分布的采样区间
    near = torch.clamp_min(gt_depths - 3 * variance, 0.0)
    far = gt_depths + 3 * variance
    dist = torch.randn((sample_rays, ray_points), device = device)
    
    if stratified is not None and int(ray_points * stratified) != 0:
        # uniform random sampling
        # 生成等距步长
        t_vals = torch.linspace(0., 1., steps = int(ray_points * stratified)).cuda()
        # 计算分层采样的的位置
        dist_strat = -3 * variance * (1.-t_vals) + 3 * variance * (t_vals)
        # 宽展维度与采样射线数量相同
        dist_strat = dist_strat.expand([sample_rays, -1])
        # 计算分层采样位置张量中每两个相邻元素的中间值
        mids = .5 * (dist_strat[...,1:] + dist_strat[...,:-1])
        # 定义上下边界，上下边界即为 dist_strat 的最后和第一列
        upper = torch.cat([mids, dist_strat[...,-1:]], -1)
        lower = torch.cat([dist_strat[...,:1], mids], -1)
        # 生成与 dist_strat 相同大小的均匀随机张量
        t_rand = torch.rand(dist_strat.shape).cuda()
        # 在上下界之间的差异进行线性插值
        dist_strat = lower + (upper - lower) * t_rand
        # 拼接两个采样张量     
        dist = torch.cat([dist, dist_strat], -1)
    
    # 对采样点进行排序
    dist = torch.sort(dist, -1).values
    
    # 对观测点附近进行采样
    # 同时限制截断距离
    z_vals = dist * variance + gt_depths
    z_vals = torch.clamp(z_vals, min = near, max = far)
    
    # （N,3）->(N,1，3)
    rays_o = rays_o.view(-1, 1, 3)
    rays_d = rays_d.view(-1, 1, 3)
    
    # 获得占据的权重值和标签
    occ_weight, label = gen_weight_label(dist, weight_coe)
    
    # 点云点的位置坐标
    xyz = rays_o + rays_d * z_vals.unsqueeze(-1)
    xyz = xyz.view(-1, 3)
    
    occ_weight = occ_weight.view(-1)
    label = label.view(-1)
    
    # 得到占据值
    model_out = nerf_model(xyz)
    occ = model_out.squeeze(-1)
    
    # model out represent occupancy
    # TODO: 公式的详细推导
    # 得到了占据网格的深度值
    alphas = model_out.view(-1, dist.shape[-1])
    
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
    
    weights = \
        alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)

    depths = torch.sum(weights * z_vals, 1)
    
    return occ, label, occ_weight, depths

def render_normal(rays, gt_depths, nerf_model, epsilon):
    sample_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    device = rays.device
    
    xyz = rays_o + rays_d * gt_depths
    xyz = xyz.view(-1, 3)
    
    xyz_neighbor = xyz + torch.randn_like(xyz, device = device) * epsilon
    
    xyz_full = torch.concat([xyz, xyz_neighbor], 0)
    
    gradient_full = nerf_model.gradient(xyz_full)
    gradient_full = gradient_full.squeeze(1) / torch.norm(gradient_full, dim = -1)
    gradient = gradient_full[:sample_rays]
    gradient_neighbor = gradient_full[sample_rays:]
    
    return gradient, gradient_neighbor