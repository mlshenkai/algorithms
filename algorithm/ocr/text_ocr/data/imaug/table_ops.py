import cv2
import numpy as np


class GenTableMask(object):
    """gen table mask"""

    def __init__(self, shrink_h_max, shrink_w_max, mask_type=0, **kwargs):
        self.shrink_h_max = 5
        self.shrink_w_max = 5
        self.mask_type = mask_type

    def projection(self, erosion, h, w, spilt_threshold=0):
        # 水平投影
        projection_map = np.ones_like(erosion)
        project_val_array = [0 for _ in range(0, h)]

        for j in range(0, h):
            for i in range(0, w):
                if erosion[j, i] == 255:
                    project_val_array[j] += 1
        # 根据数组，获取切割点
        start_idx = 0  # 记录进入字符区的索引
        end_idx = 0  # 记录进入空白区域的索引
        in_text = False  # 是否遍历到了字符区内
        box_list = []
        for i in range(len(project_val_array)):
            if in_text == False and project_val_array[i] > spilt_threshold:  # 进入字符区了
                in_text = True
                start_idx = i
            elif project_val_array[i] <= spilt_threshold and in_text == True:  # 进入空白区了
                end_idx = i
                in_text = False
                if end_idx - start_idx <= 2:
                    continue
                box_list.append((start_idx, end_idx + 1))

        if in_text:
            box_list.append((start_idx, h - 1))
        # 绘制投影直方图
        for j in range(0, h):
            for i in range(0, project_val_array[j]):
                projection_map[j, i] = 0
        return box_list, projection_map

    def projection_cx(self, box_img):
        box_gray_img = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
        h, w = box_gray_img.shape
        # 灰度图片进行二值化处理
        ret, thresh1 = cv2.threshold(box_gray_img, 200, 255, cv2.THRESH_BINARY_INV)
        # 纵向腐蚀
        if h < w:
            kernel = np.ones((2, 1), np.uint8)
            erode = cv2.erode(thresh1, kernel, iterations=1)
        else:
            erode = thresh1
        # 水平膨胀
        kernel = np.ones((1, 5), np.uint8)
        erosion = cv2.dilate(erode, kernel, iterations=1)
        # 水平投影
        projection_map = np.ones_like(erosion)
        project_val_array = [0 for _ in range(0, h)]

        for j in range(0, h):
            for i in range(0, w):
                if erosion[j, i] == 255:
                    project_val_array[j] += 1
        # 根据数组，获取切割点
        start_idx = 0  # 记录进入字符区的索引
        end_idx = 0  # 记录进入空白区域的索引
        in_text = False  # 是否遍历到了字符区内
        box_list = []
        spilt_threshold = 0
        for i in range(len(project_val_array)):
            if in_text == False and project_val_array[i] > spilt_threshold:  # 进入字符区了
                in_text = True
                start_idx = i
            elif project_val_array[i] <= spilt_threshold and in_text == True:  # 进入空白区了
                end_idx = i
                in_text = False
                if end_idx - start_idx <= 2:
                    continue
                box_list.append((start_idx, end_idx + 1))

        if in_text:
            box_list.append((start_idx, h - 1))
        # 绘制投影直方图
        for j in range(0, h):
            for i in range(0, project_val_array[j]):
                projection_map[j, i] = 0
        split_bbox_list = []
        if len(box_list) > 1:
            for i, (h_start, h_end) in enumerate(box_list):
                if i == 0:
                    h_start = 0
                if i == len(box_list):
                    h_end = h
                word_img = erosion[h_start : h_end + 1, :]
                word_h, word_w = word_img.shape
                w_split_list, w_projection_map = self.projection(
                    word_img.T, word_w, word_h
                )
                w_start, w_end = w_split_list[0][0], w_split_list[-1][1]
                if h_start > 0:
                    h_start -= 1
                h_end += 1
                word_img = box_img[h_start : h_end + 1 :, w_start : w_end + 1, :]
                split_bbox_list.append([w_start, h_start, w_end, h_end])
        else:
            split_bbox_list.append([0, 0, w, h])
        return split_bbox_list

    def shrink_bbox(self, bbox):
        left, top, right, bottom = bbox
        sh_h = min(max(int((bottom - top) * 0.1), 1), self.shrink_h_max)
        sh_w = min(max(int((right - left) * 0.1), 1), self.shrink_w_max)
        left_new = left + sh_w
        right_new = right - sh_w
        top_new = top + sh_h
        bottom_new = bottom - sh_h
        if left_new >= right_new:
            left_new = left
            right_new = right
        if top_new >= bottom_new:
            top_new = top
            bottom_new = bottom
        return [left_new, top_new, right_new, bottom_new]

    def __call__(self, data):
        img = data["image"]
        cells = data["cells"]
        height, width = img.shape[0:2]
        if self.mask_type == 1:
            mask_img = np.zeros((height, width), dtype=np.float32)
        else:
            mask_img = np.zeros((height, width, 3), dtype=np.float32)
        cell_num = len(cells)
        for cno in range(cell_num):
            if "bbox" in cells[cno]:
                bbox = cells[cno]["bbox"]
                left, top, right, bottom = bbox
                box_img = img[top:bottom, left:right, :].copy()
                split_bbox_list = self.projection_cx(box_img)
                for sno in range(len(split_bbox_list)):
                    split_bbox_list[sno][0] += left
                    split_bbox_list[sno][1] += top
                    split_bbox_list[sno][2] += left
                    split_bbox_list[sno][3] += top

                for sno in range(len(split_bbox_list)):
                    left, top, right, bottom = split_bbox_list[sno]
                    left, top, right, bottom = self.shrink_bbox(
                        [left, top, right, bottom]
                    )
                    if self.mask_type == 1:
                        mask_img[top:bottom, left:right] = 1.0
                        data["mask_img"] = mask_img
                    else:
                        mask_img[top:bottom, left:right, :] = (255, 255, 255)
                        data["image"] = mask_img
        return data


class ResizeTableImage(object):
    def __init__(self, max_len, resize_bboxes=False, infer_mode=False, **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len
        self.resize_bboxes = resize_bboxes
        self.infer_mode = infer_mode

    def __call__(self, data):
        img = data["image"]
        height, width = img.shape[0:2]
        ratio = self.max_len / (max(height, width) * 1.0)
        resize_h = int(height * ratio)
        resize_w = int(width * ratio)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        if self.resize_bboxes and not self.infer_mode:
            data["bboxes"] = data["bboxes"] * ratio
        data["image"] = resize_img
        data["src_img"] = img
        data["shape"] = np.array([height, width, ratio, ratio])
        data["max_len"] = self.max_len
        return data


class PaddingTableImage(object):
    def __init__(self, size, **kwargs):
        super(PaddingTableImage, self).__init__()
        self.size = size

    def __call__(self, data):
        img = data["image"]
        pad_h, pad_w = self.size
        padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        height, width = img.shape[0:2]
        padding_img[0:height, 0:width, :] = img.copy()
        data["image"] = padding_img
        shape = data["shape"].tolist()
        shape.extend([pad_h, pad_w])
        data["shape"] = np.array(shape)
        return data