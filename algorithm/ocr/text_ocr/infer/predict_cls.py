# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 7:47 PM
# @File: predict_cls
# @Email: mlshenkai@163.com
import torch
import time

from algorithm.ocr.text_ocr.data.imaug import create_operators
from utils.yaml_utils import Config
from algorithm.ocr.text_ocr.modeling.architectures import build_model
from algorithm.ocr.text_ocr.postprocess import build_post_process
from algorithm.ocr.utils.util import update_rec_head_out_channels, transform
import numpy as np
import cv2
import os


class TextClassifier:
    def __init__(self, config, device):
        if isinstance(config, str) and os.path.exists(config):
            self.root_path = config
            self.config = Config(os.path.join(self.root_path, "config.yaml")).cfg
        else:
            self.config = config
        infer_config = self.config["Infer"]
        self.device = device
        global_config = self.config['Global']

        # build post process
        self.post_process_class = build_post_process(self.config['PostProcess'])
        update_rec_head_out_channels(self.config, self.post_process_class)
        architecture_config = self.config["Architecture"]
        model = build_model(architecture_config)
        checkpoint = torch.load(
            os.path.join(self.root_path, infer_config["model_name"])
        )["state_dict"]
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model.to(self.device)
        transforms = build_cls_process(self.config)
        global_config['infer_mode'] = True
        self.ops = create_operators(transforms)
        self.cls_batch_num = 256
        self.cls_thresh = 0.5

    def __call__(self, img_list):
        img_num = len(img_list)
        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        with torch.no_grad():
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                tic = time.time()
                for ino in range(beg_img_no, end_img_no):
                    data = {'image': img_list[ino]}
                    norm_img = transform(data, self.ops)[0]
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                norm_img_batch = np.concatenate(norm_img_batch)
                norm_img_batch_tensor = torch.from_numpy(norm_img_batch).to(self.device)

                preds = self.model(norm_img_batch_tensor)
                # if len(preds) == 1:
                #     preds = preds[0]

                cls_result = self.post_process_class(preds)
                elapse += time.time() - tic

                for rno in range(len(cls_result)):
                    label, score = cls_result[rno]
                    cls_res[beg_img_no + rno] = [label, score]
                    if '180' in label and score > self.cls_thresh:
                        img_list[beg_img_no + rno] = cv2.rotate(img_list[beg_img_no + rno], 1)
        return img_list, cls_res, elapse

def build_cls_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms'][1:]:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        elif op_name == "SSLRotateResize":
            op[op_name]["mode"] = "test"
        transforms.append(op)
    return transforms


# if __name__ == "__main__":
#     from pyrootutils import pyrootutils
#     project_path = pyrootutils.setup_root(
#         __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
#     )
#     config = Config(f"{project_path}/resources/models/text_cls/config.yaml").cfg
#     device = torch.device("cuda")
#
#     text_cls = TextClassifier(config, device)
#     img = cv2.imread(f"{project_path}/test/data/word_4.jpg")
#     print(text_cls([img, img, img]))

