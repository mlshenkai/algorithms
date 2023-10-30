# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 4:50 PM
# @File: predict_rec
# @Email: mlshenkai@163.com
import torch
import time
import math
import cv2
import numpy as np
from algorithm.ocr.text_ocr.data.imaug import create_operators
from algorithm.ocr.text_ocr.modeling.architectures import build_model
from algorithm.ocr.text_ocr.postprocess import build_post_process
from algorithm.ocr.utils.util import update_rec_head_out_channels
from utils.yaml_utils import Config
import os




class TextRecognizer:
    def __init__(self, config, device, language: str = "ch"):
        if isinstance(config, str) and os.path.exists(config):
            self.root_path = config
            self.config = Config(os.path.join(self.root_path, f"{language}_config.yaml")).cfg
        else:
            self.config = config
        infer_config = self.config["Infer"]

        self.config["Global"]["character_dict_path"] = os.path.join(
            self.root_path, infer_config["chat_dict_file_name"]
        )
        self.config["PostProcess"]["character_dict_path"] = os.path.join(
            self.root_path, infer_config["chat_dict_file_name"]
        )

        self.post_process_class = build_post_process(self.config["PostProcess"])
        update_rec_head_out_channels(self.config, self.post_process_class)
        self.device = device
        architecture_config = self.config["Architecture"]
        global_config = self.config["Global"]
        model = build_model(architecture_config)
        model_path = os.path.join(self.root_path, infer_config["model_name"])
        checkpoint = torch.load(
            model_path
        )["state_dict"]
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model.to(self.device)
        self.transforms = self.build_rec_process(self.config)
        global_config["infer_mode"] = True
        self.ops = create_operators(self.transforms, global_config)
        self.post_process_op = build_post_process(self.config["PostProcess"])
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 128

    def __call__(self, img_list: list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        with torch.no_grad():
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                imgC, imgH, imgW = self.rec_image_shape[:3]
                max_wh_ratio = imgW / imgH
                # max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                for ino in range(beg_img_no, end_img_no):
                    norm_img = self.resize_norm_img(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                norm_img_batch = np.concatenate(norm_img_batch)
                # norm_img_batch = norm_img_batch.copy()
                norm_img_batch_tensor = torch.from_numpy(norm_img_batch).to(self.device)

                preds = self.model(norm_img_batch_tensor)
                preds = preds["res"]
                # if len(preds) == 1:
                #     preds = preds[0]
                rec_result = self.post_process_op({"res": preds})
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res, time.time() - st

    @staticmethod
    def build_rec_process(cfg):
        transforms = []
        for op in cfg["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            elif op_name in ["RecResizeImg"]:
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                if cfg["Architecture"]["algorithm"] == "SRN":
                    op[op_name]["keep_keys"] = [
                        "image",
                        "encoder_word_pos",
                        "gsrm_word_pos",
                        "gsrm_slf_attn_bias1",
                        "gsrm_slf_attn_bias2",
                    ]
                elif cfg["Architecture"]["algorithm"] == "SAR":
                    op[op_name]["keep_keys"] = ["image", "valid_ratio"]
                elif cfg["Architecture"]["algorithm"] == "RobustScanner":
                    op[op_name]["keep_keys"] = ["image", "valid_ratio", "word_positons"]
                else:
                    op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)
        return transforms

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


# if __name__ == "__main__":
#     from pyrootutils import pyrootutils
#     from algorithm.ocr.text_ocr.utils.ckpt import load_ckpt
#     from algorithm.ocr.text_ocr.utils.visual import draw_det
#
#     project_path = pyrootutils.setup_root(
#         __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
#     )
#
#     device = torch.device("cuda")
#     config = Config(f"{project_path}/resources/models/text_rec/config.yaml").cfg
#     config["Global"][
#         "character_dict_path"
#     ] = f"{project_path}/resources/models/text_rec/ocr_keys_v1.txt"
#     config["PostProcess"][
#         "character_dict_path"
#     ] = f"{project_path}/resources/models/text_rec/ocr_keys_v1.txt"
#     text_rec = TextRecognizer(config, device)
#     img = cv2.imread(f"{project_path}/test/data/word_3.jpg")
#     print(text_rec([img]))
