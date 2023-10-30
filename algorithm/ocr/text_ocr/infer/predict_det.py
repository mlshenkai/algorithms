# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 10:46 AM
# @File: predict_det
# @Email: mlshenkai@163.com
import cv2

from algorithm.ocr.text_ocr.data.imaug import create_operators, transform
from algorithm.ocr.text_ocr.engine.onnx_engine import ONNXEngine
import numpy as np
import os
import time
import torch
from algorithm.ocr.text_ocr.modeling.architectures import build_model
from algorithm.ocr.text_ocr.postprocess import build_post_process
from utils.yaml_utils import Config


class TextDetector:
    def __init__(self, config, device):
        if isinstance(config, str) and os.path.exists(config):
            self.root_path = config
            self.config = Config(os.path.join(self.root_path, "config.yaml")).cfg
        else:
            self.config = config
        self.device = device
        infer_config = self.config["Infer"]
        global_config = self.config["Global"]
        architecture_config = self.config["Architecture"]

        model_path = os.path.join(self.root_path, infer_config["model_name"])
        model = build_model(architecture_config)
        checkpoint = torch.load(
            model_path
        )["state_dict"]
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model.to(self.device)
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": self.config["det_limit_side_len"],
                    "limit_type": self.config["det_limit_type"],
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        self.preprocess_op = create_operators(pre_process_list)
        self.post_process_class = build_post_process(self.config["PostProcess"])

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        image_shape = img.shape
        data = {"image": img}
        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        images = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        images = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            preds = self.model(images)
        post_result = self.post_process_class(preds, [-1, shape_list])
        dt_boxes = post_result[0]["points"]

        # if self.args.det_box_type == "poly":
        #     dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        # else:
        #     dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, image_shape)
        et = time.time()
        return dt_boxes, et - st

#
# if __name__ == "__main__":
#     from pyrootutils import pyrootutils
#     from algorithm.ocr.text_ocr.utils.ckpt import load_ckpt
#     from algorithm.ocr.text_ocr.utils.visual import draw_det
#     project_path = pyrootutils.setup_root(
#         __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
#     )
#     device = torch.device("cuda")
#     config = Config(f"{project_path}/resources/models/text_det/config.yaml").cfg
#     text_detect = TextDetector(config, device)
#     img = cv2.imread(f"{project_path}/test/data/d14606a97f324678bf56dd7b613d5f83.jpg")
#     img_copy = img.copy()
#     dt_boxes, _ = text_detect(img)
#
#     draw_img = draw_det(dt_boxes, img_copy)
#     cv2.imwrite(f"{project_path}/test/data/det_img.jpg", draw_img)



