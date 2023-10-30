# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/16 9:15 AM
# @File: predict_ocr
# @Email: mlshenkai@163.com
import os.path

import cv2

from algorithm.ocr.text_ocr.predict_text_ocr import TextOCr
from algorithm.ocr.text_ocr.utils.visual import draw_system


class Ocr:
    def __init__(
        self,
        ocr_config: dict,
        barcode_config: dict = None,
        use_cls: bool = True,
        use_barcode: bool = False,
    ):
        self.ocr_config = ocr_config
        self.barcode_config = barcode_config
        self.use_cls = use_cls
        self.use_barcode = use_barcode
        self.text_ocr = TextOCr(ocr_config=self.ocr_config, use_cls=self.use_cls)

    def predict(self, image_numpy, save_res: bool=False, save_result_dir: str = None, language: str = "ch"):
        text_boxes, text_res, time_dict = self.text_ocr(image_numpy, language=language)
        if self.use_barcode:
            pass
        if save_res:
            from pyrootutils import pyrootutils

            project_path = pyrootutils.setup_root(
                __file__,
                project_root_env_var=True,
                dotenv=True,
                pythonpath=True,
                cwd=False,
            )
            text_list = [text_res[i][0] for i in range(len(text_res))]
            scores = [text_res[i][1] for i in range(len(text_res))]
            draw_img = draw_system(
                image_numpy,
                text_boxes,
                text_list,
                scores,
                font_path=f"{project_path}/resources/models/text_rec/simfang.ttf",
            )
            cv2.imwrite(os.path.join(save_result_dir, "ocr.jpg"), draw_img)
        return text_boxes, text_res, time_dict


if __name__ == "__main__":
    from pyrootutils import pyrootutils
    project_path = pyrootutils.setup_root(
        __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = {
        "save_crop_res": False,
        "crop_res_save_dir": "",
        "text_det_config": f"{project_path}/resources/models/text_det",
        "text_cls_config": f"{project_path}/resources/models/text_cls",
        "text_reg_config": f"{project_path}/resources/models/text_rec",
    }
    text_ocr = Ocr(ocr_config=config, use_cls=True)
    import cv2

    img = cv2.imread(f"{project_path}/test/data/d14606a97f324678bf56dd7b613d5f83.jpg")
    text_ocr.predict(img, save_res = True, save_result_dir = f"{project_path}/test/data")
