# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.tscale = opt["temporal_scale"]  #100
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]  # 0.5 不知道干嘛的
        self.num_sample = opt["num_sample"]  # 32
        self.num_sample_perbin = opt["num_sample_perbin"]  # 3
        self.feat_dim=opt["feat_dim"]  # 400

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # 构建sampling mask权重矩阵
        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )  #400-->256  256-->256

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),  # 256-->256
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),   #256-->1
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),  # 256-->256
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),  #256-->1
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),   # 256-->256
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1)),   # 256-->512
            nn.ReLU(inplace=True)
        )
        
        # 用2D卷积来捕获更多的上下文
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),  # 512-->128
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),  # 128-->128
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),  # 128-->128
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),  # 128-->2
            nn.Sigmoid()
        )

    def forward(self, x):  #(16,400,100)
        base_feature = self.x_1d_b(x)  # (16,256,100)
        start = self.x_1d_s(base_feature).squeeze(1)  #(16,100)
        end = self.x_1d_e(base_feature).squeeze(1)  #(16,100)
        
        confidence_map = self.x_1d_p(base_feature)  # (16,256,100)
        confidence_map = self._boundary_matching_layer(confidence_map)  # (16,256,32,100,100) 
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)  #(16,512,100,100)
        confidence_map = self.x_2d_p(confidence_map)  #(16,2,100,100)
        return confidence_map, start, end
    
    
    # match_layer的过程可以这么理解，对于256x100的特征矩阵，某个预设提议对应的mask权重矩阵为100x32
    # 通过两个矩阵的点乘实现了插值操作。具体来说，对于特征矩阵的某一行(1x100)和mask权重矩阵的某一列(100x1)实质上是对特征矩阵进行了加权求和操作
    def _boundary_matching_layer(self, x):  #(16,256,100)
        input_size = x.size()
        # (16,256,100)x(100,320000)-->(16,256,320000)-->(16,256,32,100,100)
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out
    
    # 对于某一个提议，将该提议等间距划分为32*3 = 96个采样点，然后每三个采样点作为一个bin
    # 对于bin中的每一个采样点，分别对采样点时序位置进行向上和向下取整(既可以获得整数的位置)，对于每个位置按照论文中的公式1计算mask权重
    # seg_xmin = -0.5  seg_xmax = 0.5  tscale= 100 num_sample=32  num_sample_perbin = 3
    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        # 这一部分对应了论文中的公式(1)
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                # 这两步其实就是对采样点分别向上和向下取整
                sample_upper = math.ceil(sample)  # 向上取整
                # modf() 方法返回x的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示。
                # 例如：math.modf(100.12) :  (0.12000000000000455, 100.0)
                sample_decimal, sample_down = math.modf(sample)  #向下取整
                # 如果采样点位于[0-99],那么左侧的值为1-小数，右侧的值为小数
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector  #(100,)
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)  # (100,32)  按列拼接
        return p_mask

   
    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.tscale):  # proposal的起始时间
            mask_mat_vector = []
            for duration_index in range(self.tscale):  #proposal的持续时间
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1  # 提议时长
                    # 拓展提议的边界
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    # self.num_sample = 32  self.num_sample_perbin = 3
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)  #(100,32)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)  # (100,32,100)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)  # (100,32,100,100) 每个时刻每种时长的提议采样32个点
        mask_mat = mask_mat.astype(np.float32)
        
        # 这就是网络主要要学习的东西,sample_mask的本质起始就是对所有可能的提议截取固定长度的特征并对特征进行加权
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)  #(100,320000)


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    model=BMN(opt)
    input=torch.randn(2,400,100)
    a,b,c=model(input)
    print(a.shape,b.shape,c.shape)
    
    
    
    
