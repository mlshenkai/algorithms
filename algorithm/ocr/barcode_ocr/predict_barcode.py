# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 8:56 PM
# @File: predict_barcode
# @Email: mlshenkai@163.com

import os
import sys
import numpy as np
import time
import math
import cv2
import torch
from algorithm.ocr.barcode_ocr import build_model
from algorithm.ocr.barcode_ocr.post_processing import get_post_processing
from torchvision import transforms




class BarCodeDetector:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        model_path = self.config["model_path"]
        model_config = self.config['Architecture']

        self.model = build_model("Model", **model_config).to(self.device)

        self.post_process = get_post_processing(
            self.config['PostProcess']
        )
        # self.post_process.box_thresh = self.config["post_p_thre"]
        self.img_mode = "RGB"
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # 转为Tensor 归一化至0～1
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # 归一化
            ]
        )

    def __call__(self, img, is_output_polygon=False, min_scale=480, max_scale=480):
        """
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        """
        st = time.time()
        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, min_scale, max_scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.cuda()
        batch = {"shape": [(h, w)]}
        with torch.no_grad():
            preds = self.model(tensor)
            # print('==preds.shape:', preds.shape)#(b,2,h,w)
            box_list, score_list = self.post_process(
                batch, preds, is_output_polygon=is_output_polygon
            )
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = (
                        box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
                    )  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
        et = time.time()
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, et - st


def resize_image(img, min_scale=480, max_scale=480):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(min_scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_scale:
        im_scale = float(max_scale) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 32 == 0 else (new_h // 32 + 1) * 32
    new_w = new_w if new_w // 32 == 0 else (new_w // 32 + 1) * 32
    # print('==new_h,new_w:', new_h, new_w)
    re_im = cv2.resize(img, (new_w, new_h))
    return re_im


if __name__ == "__main__":
    from pyrootutils import pyrootutils
    from utils.yaml_utils import Config
    from algorithm.ocr.text_ocr.utils.visual import draw_det
    project_path = pyrootutils.setup_root(
        __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )
    device = torch.device("cuda")
    config = Config(f"{project_path}/resources/models/barcode_det/config.yaml").cfg
    config["model_path"] = f"{project_path}/resources/models/barcode_det/best_accuracy.pth"
    barcode_det = BarCodeDetector(config, device)
    print(barcode_det)
    img = cv2.imread(f"{project_path}/test/data/express_img_15077229_0.jpg")
    img_origin = img.copy()
    preds, boxes_list, score_list, time_cost = barcode_det(img)
    preds, boxes_list, score_list, time_cost = barcode_det(img)
    preds, boxes_list, score_list, time_cost = barcode_det(img)
    print(time_cost)
    print(boxes_list)
    draw_img = draw_det(boxes_list, img_origin)
    cv2.imwrite(f"{project_path}/test/data/barcode_img.jpg", draw_img)
    cv2.imwrite(f"{project_path}/test/data/barcode_imgPred.jpg", preds*255)

