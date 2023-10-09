import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, N_freqs = 8, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

def hashfunction(coords, log2_hashmap_size):
    '''
    coords: 3D coordinates. B x 3
    log2T:  logarithm of T w.r.t 2
    '''
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    # 原文公式（4），对各个位置坐标乘一个大质数后进行异或操作
    # & 前面部分是得到 2^log2_hashmap_size 位的全为一的掩码
    # 用来确保哈希值落在 0 ~ 2^log2_hashmap_size -1 之间
    return ((1<<log2_hashmap_size)-1) & (x*73856093 ^ y*19349663 ^ z*83492791)

BOX_OFFSETS = torch.tensor([[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]],
                               device='cuda')

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    # 得到当前分辨率下最大边界的左下顶点和右上顶点
    box_min, box_max = bounding_box

    # 判断是否所有采样点在 box 中
    # 如果不满足上述条件，则用 torch.clamp 对超出范围的点做截断处理
    # 截断范围定义在 min 和 max
    if not torch.all(xyz < box_max) or not torch.all(xyz > box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    # 求得每个网格的大小
    grid_size = (box_max-box_min)/resolution
    # 向下取整，得到采样点左下角顶点的位置坐标,即在 box 坐标系下的位置索引
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    # 得到采样点所在体素网格的左下角顶点位置，即在 box 坐标系下的实际位置
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    # 得到采样点所在体素网格的右上角顶点位置，即在 box 坐标系下的实际位置
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0]).cuda()*grid_size
    # 得到采样点所在体素的所有八个顶点位置
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    # 得到体素各个顶点的 hash 编码值
    hashed_voxel_indices = hashfunction(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

def trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: B x 3
    voxel_min_vertex: B x 3
    voxel_max_vertex: B x 3
    voxel_embedds: B x 8 x 2
    '''
    # 对采样点进行三次线性插值
    # 插值是在各个顶点得到的特征向量间进行插值
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

    # step 1
    # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
    c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
    c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
    c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
    c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

    # step 2
    c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
    c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

    # step 3
    c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

    return c

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        # Instant-ngp hash 编码，所选参数与 ngp 中消融实验得到的 trade off 参数相同
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        # 输出维度为 hash 层数 * 每一层得到的特征向量的维度
        self.out_dim = self.n_levels * self.n_features_per_level
        # Instant-ngp 中的每层之间的步长 b
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        # 通过 nn.Embedding 将离散的正说索引映射到连续的向量表示
        # 依次生成了 n_levels 个嵌入层，对应到每个分辨率下的 hash 表
        # 最后通过 nn.ModuleList 将多个子模块整合管理
        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        # 对嵌入层的权重进行均匀分布的初始化
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            # 原文中公式（2），得到每一层的分辨率大小
            resolution = torch.floor(self.base_resolution * self.b**i)
            # 得到采样点所在体素的最小顶点、最大顶点以及体素八个顶点计算的哈希值
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            # 查询每一个嵌入层哈希值对应的索引参数
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            # 对采样点进行体素特征三次线性插值
            x_embedded = trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            # 添加进插值后的列表中
            x_embedded_all.append(x_embedded)
        # 对插值的特征向量进行拼接
        return torch.cat(x_embedded_all, dim=-1)

class GridEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels = 2, n_features = 32, resoluton = [64, 256]):
        super(GridEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.resoluton = resoluton
        self.n_features = n_features
        self.n_levels = n_levels

        self.embeddings = nn.ModuleList([nn.Embedding(resoluton[i] * resoluton[i] * resoluton[i], 
                                        self.n_features) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)
