# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 10:49 AM
# @File: onnx_engine
# @Email: mlshenkai@163.com
import os
import onnxruntime


class ONNXEngine:
    def __init__(self, onnx_model_path: str, use_gpu: bool = False):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"model file not found {onnx_model_path}")

        if use_gpu:
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self.onnx_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=providers
        )

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    @staticmethod
    def get_input_name(onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    @staticmethod
    def get_output_name(onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    @staticmethod
    def get_input_feed(input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def run(self, image_numpy):
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result


if __name__ == "__main__":
    from pyrootutils import pyrootutils
    project_path = pyrootutils.setup_root(
        __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )
    import cv2
    import time

    det_onnx_engine = ONNXEngine(f"{project_path}/resources/models/text_det/text_det_infer.onnx", True)
    img = cv2.imread(f"{project_path}/test/data/20230904113414.jpg")
    start = time.time()
    print(det_onnx_engine.run(img))
    print(time.time() - start)