# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 9:37 AM
# @File: predict_ocr
# @Email: mlshenkai@163.com
from pyrootutils import pyrootutils
import torch
import numpy as np
import time
import os


project_path = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)

from algorithm.ocr.text_ocr.utils.utility import (
    get_minarea_rect_crop,
    get_rotate_crop_image,
    check_and_read,
    get_image_file_list,
    check_gpu,
)
from algorithm.ocr.text_ocr.infer import TextDetector, TextRecognizer, TextClassifier
from utils.yaml_utils import Config


class TextOCr:
    def __init__(self, ocr_config: dict, use_cls: bool = True):
        self.args = ocr_config
        text_det_config = ocr_config["text_det_config"]
        text_cls_config = ocr_config["text_cls_config"]
        text_reg_config = ocr_config["text_reg_config"]
        if check_gpu(True):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.text_detector = TextDetector(config=text_det_config, device=self.device)
        self.text_classifier = TextClassifier(
            config=text_cls_config, device=self.device
        )
        self.ch_text_recognizer = TextRecognizer(
            config=text_reg_config, device=self.device
        )
        self.en_text_recognizer = TextRecognizer(
            config=text_reg_config, device=self.device, language="en"
        )
        self.crop_image_res_index = 0
        self.use_angle_cls = use_cls

    def __call__(self, image: np.ndarray, language: str = "ch"):
        start = time.time()
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}
        dt_boxes, det_elapse = self.text_det(image)
        time_dict["det"] = det_elapse
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            # tmp_box = copy.deepcopy(dt_boxes[bno])
            tmp_box = dt_boxes[bno]
            # img_crop = get_rotate_crop_image(image, tmp_box)
            # if self.args.det_box_type == "quad":
            #     img_crop = get_rotate_crop_image(image, tmp_box)
            # else:
            #    img_crop = get_minarea_rect_crop(image, tmp_box)
            img_crop = get_minarea_rect_crop(image, tmp_box)
            img_crop_list.append(img_crop)

        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict["cls"] = elapse
        if language == "ch":
            rec_res, rec_elapse = self.ch_text_recognizer(img_crop_list)
        else:
            rec_res, rec_elapse = self.en_text_recognizer(img_crop_list)
        time_dict["rec"] = rec_elapse
        if self.args["save_crop_res"]:
            self.draw_crop_rec_res(
                self.args["crop_res_save_dir"], img_crop_list, rec_res
            )
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= 0.5:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict

    def text_det(self, image: np.ndarray):
        dt_boxes, elapse = self.text_detector(image)
        return dt_boxes, elapse

    def text_cls(self, img_list: list):
        img_list, cls_res, elapse = self.text_classifier(img_list)
        return img_list, cls_res, elapse

    def text_rec(self, img_list: list):
        rec_res, elapse = self.text_recognizer(img_list)
        return rec_res, elapse

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )
            # logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


# if __name__ == "__main__":
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     config = {
#         "save_crop_res": False,
#         "crop_res_save_dir": "",
#         "text_det_config": f"{project_path}/resources/models/text_det",
#         "text_cls_config": f"{project_path}/resources/models/text_cls",
#         "text_reg_config": f"{project_path}/resources/models/text_rec",
#     }
#     text_ocr = TextOCr(ocr_config=config, use_cls=True)
#     import cv2
#
#     img = cv2.imread(f"{project_path}/test/data/express_img_15063753_0.jpg")
#     filter_boxes, filter_rec_res, time_dict = text_ocr(img)
#     # print(filter_rec_res)
#     # print(time_dict)
#     filter_boxes, filter_rec_res, time_dict = text_ocr(img)
#     print(filter_rec_res)
#     print(time_dict)
