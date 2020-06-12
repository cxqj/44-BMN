# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
from utils import ioa_with_anchors, iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset  #train/validation
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()
        self._get_match_map()
        
    # 获取标注字典
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]  # duration_second, duration_frame,annotations, feature_frame
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))  # train:9649  test:4728

    def __getitem__(self, index):
        video_data = self._load_file(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data,confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _get_match_map(self):
        match_map = []
        # 遍历每一个时序位置
        for idx in range(self.temporal_scale):
            tmp_match_window = []  # [[0.0,0.01],[0.0,0.02],.....,[0.0,1.0]]
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2  100个时序位置
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        """
        match_map数据格式（所有列的起始时间相同，所有行的持续时间相同）:
             0.0       0.01
             0.01      0.02
             0.02      0.03
               :        :
             0.99      1.0
          ---------------------
             0.0       0.02
             0.01      0.03
             0.02      0.04
               :        :
             0.99      1.01
         ----------------------
             0.0       0.03
             0.01      0.04
             0.02      0.05
               :        :
             0.99      1.02
          ---------------------
               ........
          ---------------------
             0.0       1.0
             0.01      1.01
             0.02      1.02
               :        :
             0.99      1.99
          ---------------------
        """
        
        self.match_map = match_map  # duration is same in row, start is same in col  (10000,2) 
        # 间隔0.02
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]         # [-0.005,0.005,0.015,....,0.985]
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]  # [0.015,0.025,....1.005]

    # 加载视频特征
    def _load_file(self, index):
        video_name = self.video_list[index]
        video_df = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name + ".csv")
        video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data
    
    
    # 获取预设anchor_min和anchor_max与gt_start区域和gt_end区域的重叠率和每一种anchor的iou_map
    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']   #1128
        video_second = video_info['duration_second']  #47.114
        feature_frame = video_info['feature_frame']  # 1120
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used  46.77
        video_labels = video_info['annotations']  # the measurement is second, not frame  [{'segment':[0.01,37.11],'labels':'waxing skis'}]

        ##############################################################################################
        # change the measurement from second to percentage
        # 计算的是起止时间相对于视频时长的百分比
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)  # 0.00
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)  # 0.79
            gt_bbox.append([tmp_start, tmp_end])
            # 计算当前gt_bbox与所有预设anchor的IOU
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)  #(10000,)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_scale, self.temporal_scale])  #(100,100)
            gt_iou_map.append(tmp_gt_iou_map)
            
        # 相当于建立了一个字典保存所有可能的提议与gt_bbox的IOU值
        gt_iou_map = np.array(gt_iou_map)  # (1,100,100) 其中1表示gt_bbox的个数
        gt_iou_map = np.max(gt_iou_map, axis=0)  # 如果存在多个gt_bbox，则选取最大IOU值作为iou_map中的值
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        # 将gt的起止时间扩大为一个范围
        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        # 间隔0.03
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)  # [0.12,0.15]
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)  # [0.85,0.88]
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        # 计算每个0.02的小区间与gt_start和gt_end的重叠度
        match_score_start = []  # (100)
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []   # (100)
        
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a,b,c,d in train_loader:
        print(a.shape,b.shape,c.shape,d.shape)
        break
