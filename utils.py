import argparse
import json
import os
import shutil
from datetime import datetime
from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class Tool:
    def __int__(self):
        super(Tool, self).__int__()

    @staticmethod
    def copy_file_to_dir(src1, src2, dist):
        cp_file = []
        for file in os.listdir(src1):
            # copy file from src2/mask to dist/test_label
            cp_file.append(file.split('.')[0] + '.png')
        for file in os.listdir(src2):
            if file in cp_file:
                p = os.path.join(src2, file)
                shutil.copy(p, dist)
                print('[%s]: %s copy ok!' % (datetime.now(), file))

    @staticmethod
    def json2mask(image_path, json_path, mask_path, category_types: list[str]):
        """
        把json转成mask,二分类
        :param image_path: 原图路径
        :param json_path: json文件路径
        :param mask_path: 生成的label mask路径
        :param category_types: 类别，如果是多分类，一定要有Background这一项且一定要放在index为0的位置
        """
        color = [255 for _ in range(len(category_types))]
        print(len(category_types), ' 类')
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
            # 将图片标注json文件批量生成训练所需的标签图像png
            for img_path in os.listdir(image_path):
                img_name = img_path.split('.')[0]
                img = cv2.imread(os.path.join(image_path, img_path))
                h, w = img.shape[:2]
                # 创建一个大小和原图相同的空白图像
                mask = np.zeros([h, w, 1], np.uint8)

                with open(json_path + img_name + '.json', encoding='utf-8') as f:
                    label = json.load(f)

                shapes = label['shapes']
                for shape in shapes:
                    category = shape['label']
                    points = shape['points']
                    # 将图像标记填充至空白图像
                    points_array = np.array(points, dtype=np.int32)
                    mask = cv2.fillPoly(mask, [points_array], color[category_types.index(category)])

                # 生成的标注图像必须为png格式
                save_name = mask_path + img_name + '.png'
                cv2.imwrite(save_name, mask)
                print(save_name + '\tfinished')
                # break

    @staticmethod
    def dice_coeff(imgPredict, imgLabel):
        smooth = 1.
        intersection = (imgPredict * imgLabel).sum()
        return (2. * intersection + smooth) / (imgPredict.sum() + imgLabel.sum() + smooth)

    @staticmethod
    def get_mask_color(n):
        color_map = np.zeros([n, 3]).astype(np.uint8)
        print(color_map)
        color_map[0, :] = np.array([0, 0, 0])
        color_map[1, :] = np.array([244, 35, 232])
        color_map[2, :] = np.array([70, 70, 70])
        color_map[3, :] = np.array([102, 102, 156])
        color_map[4, :] = np.array([190, 153, 153])
        color_map[5, :] = np.array([153, 153, 153])

        color_map[6, :] = np.array([250, 170, 30])
        color_map[7, :] = np.array([220, 220, 0])
        color_map[8, :] = np.array([107, 142, 35])
        color_map[9, :] = np.array([152, 251, 152])
        color_map[10, :] = np.array([70, 130, 180])

        color_map[11, :] = np.array([220, 20, 60])
        color_map[12, :] = np.array([119, 11, 32])
        color_map[13, :] = np.array([0, 0, 142])
        color_map[14, :] = np.array([0, 0, 70])
        color_map[15, :] = np.array([0, 60, 100])

        color_map[16, :] = np.array([0, 80, 100])
        color_map[17, :] = np.array([0, 0, 230])
        color_map[18, :] = np.array([255, 0, 0])
        print(color_map)

        return color_map


class Colorize:
    '''
    这里实现了输入一张mask（output）,输出一个三通道的彩图。
    '''

    def __init__(self, n=19):
        self.cmap = Tool.get_mask_color(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])  # array->tensor

    def __call__(self, gray_image):
        size = gray_image.size()  # 这里就是上文的output
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image == label
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def split_raw(big_raw_img, imgsz=640, overlap_size=64):
    # 计算w方向的切割份数和重叠尺寸
    raw_size_hw = big_raw_img.shape[0:2]
    if raw_size_hw[0] < imgsz:
        n_row = 1
        overlap_h = 0
    else:
        for i in range(2, 20, 1):
            dh = (imgsz * i - raw_size_hw[0]) / (i - 1)
            if dh > overlap_size:
                n_row = i
                overlap_h = dh
                break

    if raw_size_hw[1] < imgsz:
        n_col = 1
        overlap_w = 0
    else:
        for j in range(2, 20, 1):
            dw = (imgsz * j - raw_size_hw[1]) / (j - 1)
            if dw > overlap_size:
                n_col = j
                overlap_w = dw
                break

    # 切割，获得每张图片的左上角点和右下角点
    top_list = []
    bottom_list = []
    for idx in range(n_row):
        top = int(round(idx * (imgsz - overlap_h)))
        down = int(round(min(imgsz + idx * (imgsz - overlap_h), raw_size_hw[0])))
        top_list.append(top)
        bottom_list.append(down)

    # 切割，获得每张图片的左上角点和右下角点
    left_list = []
    right_list = []
    for jdx in range(n_col):
        left = int(round(jdx * (imgsz - overlap_w)))
        right = int(round(min(imgsz + jdx * (imgsz - overlap_w), raw_size_hw[1])))
        left_list.append(left)
        right_list.append(right)

    all_images = []
    all_images_lt_rb = []
    real_overlap_size_wh = [overlap_w, overlap_h]
    for idx in range(n_row):
        for jdx in range(n_col):
            sub_img = big_raw_img[top_list[idx]:bottom_list[idx], left_list[jdx]:right_list[jdx]]
            all_images.append(sub_img)
            all_images_lt_rb.append((left_list[jdx], top_list[idx], right_list[jdx], bottom_list[idx]))

    return all_images, all_images_lt_rb


# 切割五通道的,注意f11标注的，可改成单通道的切割
def handle_dataset_resize(img_path, mask_path, save_img_dir, save_mask_dir):
    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    if os.path.exists(save_mask_dir):
        shutil.rmtree(save_mask_dir)
    os.mkdir(save_img_dir)
    os.mkdir(save_mask_dir)

    for tiff in os.listdir(img_path):
        p = img_path + tiff
        img = io.imread(p)
        img = np.transpose(img[:5, :, :], (1, 2, 0))
        # img = np.transpose(img, (1, 2, 0))
        # img = cv2.imread(p, 0)
        x, y = split_raw(img)

        for i, sub_img in enumerate(x):
            sub_img = np.transpose(sub_img, (2, 0, 1))
            # print(sub_img.shape)
            save_sub_img_path = save_img_dir + tiff.split('.')[0] + '_' + str(i) + '.tiff'
            # save_sub_img_path = save_img_dir + tiff.split('.')[0] + '_' + str(i) + '.bmp'
            io.imsave(save_sub_img_path, sub_img)
            # cv2.imwrite(save_sub_img_path, sub_img)
            print(f'[{datetime.now()}] ====> {save_sub_img_path} \t save ok')

    for each_mask in os.listdir(mask_path):
        p = mask_path + each_mask
        mask = cv2.imread(p, 0)
        x, y = split_raw(mask)

        for i, sub_mask in enumerate(x):
            save_sub_mask_path = save_mask_dir + each_mask.split('.')[0] + '_' + str(i) + '.png'
            cv2.imwrite(save_sub_mask_path, sub_mask)
            print(f'[{datetime.now()}]====> {save_sub_mask_path}\t save ok')


def test_split_img_and_mask():
    raw_img_path = r'D:\py_program\testAll\segement\src\dataset_dm_zm/'
    raw_mask_path = r'D:\files\data\all_mask_data/'
    new_img_path = r'D:\py_program\testAll\segement\src\data\img/'
    new_mask_path = r'D:\py_program\testAll\segement\src\data\mask/'
    handle_dataset_resize(raw_img_path, raw_mask_path, new_img_path, new_mask_path)
    print('split finished')


# 这个已经在测试代码实现，可以不执行
def resize_test_img_and_mask(
        test_img_path='./test_img/',
        true_label_path='./test_label/',
        after_resize_test_img_path='./test_resize_img/',
        after_resize_test_mask_path='./test_resize_mask/'
):
    if os.path.exists(after_resize_test_img_path):
        shutil.rmtree(after_resize_test_img_path)
    if os.path.exists(after_resize_test_mask_path):
        shutil.rmtree(after_resize_test_mask_path)
    os.mkdir(after_resize_test_mask_path)
    os.mkdir(after_resize_test_img_path)
    handle_dataset_resize(test_img_path, true_label_path, after_resize_test_img_path, after_resize_test_mask_path)
    print('resize test dataset ok')


def test_split_train_and_test(radio=0.2,
                              train_path=r'D:\py_program\testAll\segement\src\dataset_dm_zm/',
                              train_labels=r'D:\files\data\all_mask_data/',
                              test_path=r'D:\files\data\test_data/',
                              test_labels=r'D:\files\data\test_mask/'
                              ):
    """
    从总数据集中切分radio比例到测试集中
    :param radio: 测试集的占比
    """
    import os
    import shutil
    import random

    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if os.path.exists(test_labels):
        shutil.rmtree(test_labels)
    os.makedirs(test_path)
    os.makedirs(test_labels)

    all_dataset = os.listdir(train_path)
    train_data_num = int(len(all_dataset) * radio)
    train_dataset = random.sample(all_dataset, train_data_num)
    for name in train_dataset:
        shutil.move(os.path.join(train_path, name), os.path.join(test_path, name))
        shutil.move(os.path.join(train_labels, name[:-5] + '.png'), os.path.join(test_labels, name[:-5] + '.png'))
    print('split finished')


def clear_test_res():
    after_resize_test_img_path = './test_resize_img/'
    after_resize_test_mask_path = './test_resize_mask/'
    if os.path.exists(after_resize_test_img_path):
        shutil.rmtree(after_resize_test_img_path)
    if os.path.exists(after_resize_test_mask_path):
        shutil.rmtree(after_resize_test_mask_path)
    os.mkdir(after_resize_test_img_path)
    os.mkdir(after_resize_test_mask_path)


def move_file_from_dir1_to_dir2(dir1, dir2, dir3):
    '''
    :param dir1:全部都有
    :param dir2: 只有一部分
    :param dir3: dir1 - dir2
    '''
    d1, d2 = set(), handle_CAN_train_data(dir2)
    for file in os.listdir(dir1):
        d1.add(file)
    print(f'全部有 {len(d1)}张')
    d3 = d1 - d2
    for d in d3:
        shutil.move(dir1 + d, dir3 + d)
    print('移动完毕')


def handle_CAN_train_data(train_path_resized):
    train_set = set()
    for file in os.listdir(train_path_resized):
        # raw_file = file.split('.')[0][:-2] + '.bmp'
        raw_file = file.split('.')[0][:-2] + '.png'
        train_set.add(raw_file)
    print(f'train有 {len(train_set)} 张')
    return train_set


def evaluation_model(pred_img, raw_sub_img, args):
    pred_img = np.where(pred_img > args.threshold, 1, 0)
    # print(pred_img.shape, raw_sub_img.shape)
    # raw_sub_img = np.where(raw_sub_img > args.threshold, 1, 0)
    # 模型评估
    metric = SegmentationMetric(args.class_number)
    imgPredict = np.array(pred_img.reshape(-1), np.uint8)
    # imgLabel = cv2.resize(raw_sub_img, args.img_size)
    imgLabel = np.array((raw_sub_img / 255).reshape(-1), np.int8)
    # print(imgLabel.shape, imgPredict.shape)
    metric.addBatch(imgPredict, imgLabel)
    # pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy()
    # mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    # dice = mc.binary.dc(imgPredict, imgLabel)

    # print('m_dice:', dice)
    # print('像素准确率 PA is : %f' % pa)
    # print('类别像素准确率 CPA is :', cpa)
    # print('类别平均像素准确率 MPA is : %f' % mpa)
    print('sub out mIoU is : %.6f' % mIoU)
    return mIoU


def paste_evaluation(img, m_iou, save_path):
    text = str(m_iou)
    left_down_location = (0, img.shape[0] - 25)
    # img = io.imread(img_path)
    # img = np.transpose(img[:5, :, :], (1, 2, 0))
    cv2.putText(img, text, left_down_location,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(255, 255, 255), thickness=3,
                lineType=None, bottomLeftOrigin=None)
    cv2.imwrite(save_path, img)


def build_dataset_path():
    train_data = r'D:\py_program\testAll\segement\src\dataset_dm_zm/'
    train_label = r'D:\py_program\testAll\segement\src\data\mask/'
    train_txt = './data_path_txt/train_path.txt'
    txt_name = train_txt.split('/')[:-1]
    txt_root = '/'.join(txt_name)
    if os.path.exists(txt_root):
        shutil.rmtree(txt_root)
        os.makedirs(txt_root)
    n = len(os.listdir(train_data))
    with open(train_txt, 'a+', encoding='utf-8') as f:
        for idx, file in enumerate(os.listdir(train_data)):
            data_path = os.path.join(train_data, file)
            label_path = os.path.join(train_label, file.split('.')[0] + '.png')
            if n == idx + 1:
                f.write(data_path + ' ' + label_path)
            else:
                f.write(data_path + ' ' + label_path + '\n')
            print(data_path + ' ' + label_path, ' ====== ok')
    print('built ok')


def test_show_diff_pred_raw():
    raw_path = r'D:\files\data\test_mask'
    pred_path = r'D:\files\data\save_img'
    raw_list, pred_list = [], []
    for raw in os.listdir(raw_path):
        img_path = os.path.join(raw_path, raw)
        img = cv2.imread(img_path, 0)
        raw_list.append(img)
    for pred in os.listdir(pred_path):
        img_path = os.path.join(pred_path, pred)
        img = cv2.imread(img_path)
        # img = io.imread(img_path)
        # img = np.transpose(img[:5, :, :], (1, 2, 0))
        pred_list.append(img)

    for i in range(len(raw_list)):
        plt.figure(figsize=(18, 9))
        # plt.figure()
        plt.subplot(1, 2, 1)
        img = plt.imshow(raw_list[i])
        img.set_cmap('gray')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_list[i])
        plt.show()
        # time.sleep(2)


def get_label_box(json_data_path):
    with open(json_data_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
        shapes = d['shapes']
        all_points_and_label, all_points = [], []

        for each_block in shapes:
            label = each_block['label']
            points = each_block['points']
            all_points.append(points)
            x_point, y_point = [], []
            for point in points:
                x_point.append(point[0])
                y_point.append(point[1])
            min_x, min_y, max_x, max_y = min(x_point), min(y_point), max(x_point), max(y_point)
            all_points_and_label.append((label, (min_x, min_y, max_x, max_y)))
        return all_points_and_label, all_points


def get_one_point_box(points: list):
    x_point, y_point = [], []
    for point in points:
        x_point.append(point[0])
        y_point.append(point[1])
    return min(x_point), min(y_point), max(x_point), max(y_point)


def show_chinese(img, text, pos):
    """
    :param img: opencv 图片
    :param text: 显示的中文字体
    :param pos: 显示位置
    :return:    带有字体的显示图片（包含中文）
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font=r'D:\files\font/PingFang-Bold-2', size=50)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    img_cv = np.array(img_pil)  # PIL图片转换为numpy
    img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # PIL格式转换为OpenCV的BGR格式
    return img


def judge_Contour_intersection(contour1, contour2, img_size=(2001, 2551, 1)):
    """
    这种求出的是轮廓的交点，没有实心交点，所以如果是pred的轮廓包含true的轮廓，
    那么这种情况就是两个轮廓没有交点，出现误判漏检，所以要加个轮廓包围轮廓的判断；
    :param contour1:img1的所有轮廓
    :param contour2:img2的所有轮廓
    :param img_size:图片尺寸，必须是3通道才可以绘制
    :return:所有交点
    """
    blank = np.zeros(img_size)
    image1 = cv2.drawContours(blank.copy(), contour1, -1, (255, 255, 255), 5)
    image2 = cv2.drawContours(blank.copy(), contour2, -1, (255, 255, 255), 5)
    intersection = np.logical_and(image1, image2)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1,2,1)
    # img = plt.imshow(intersection)
    # img.set_cmap('gray')
    #
    # plt.subplot(1,2,2)
    # img = plt.imshow(image2)
    # img.set_cmap('gray')
    #
    # plt.show()
    # exit(0)
    a = np.reshape(intersection, img_size[:2])
    w, h = a.shape[1], a.shape[0]
    intersection_points = []
    for i in range(h):
        for j in range(w):
            if a[i][j]:
                intersection_points.append((j, i))

    return intersection_points


# def check_overkill(pred_boxes, intersection_points):
#     overkill_boxes = set()
#     for box in pred_boxes:
#         for x, y in intersection_points:
#             is_in = (box[0] <= x <= box[1]) and (box[1] <= y <= box[3])
#             if is_in is False:
#                 overkill_boxes.add(box)
#     return overkill_boxes


# 检查漏检，实际上过杀反过来做即可
def draw_mistake(pred_mask_root=r'D:\files\data\save_mask/',
                 pred_img_root=r'D:\files\data\save_img/',
                 test_data_path=r'D:\files\data\test_data/',
                 test_mask_path=r'D:\files\data\test_mask/',
                 json_data_root=r'D:\work\new_data\new_zdm_json\data_zdm_last/'):
    all_mistake_nums = 0
    for file in os.listdir(test_data_path):
        # if 'V191152019F0PH41J_DM_0_20210604_134223' in file:
        pred_mask_path = os.path.join(pred_mask_root, 'test_mask_' + file.split('.')[0] + '.png')
        true_mask_path = os.path.join(test_mask_path, file.split('.')[0] + '.png')
        img_path = os.path.join(test_data_path, file)
        json_path = os.path.join(json_data_root, file.split('.')[0] + '.json')
        pred_img_path = os.path.join(pred_img_root, 'test_img_' + file)

        pred_mask = cv2.imread(pred_mask_path, 0)
        true_mask = cv2.imread(true_mask_path, 0)
        # raw_img_size = pred_mask.shape
        true_mask_boxes, true_mask_points = get_label_box(json_path)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_true, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        intersection_res = judge_Contour_intersection(contours, contours_true)

        pred_mask_points = []
        for p in contours:
            points = p.reshape(-1, 2).tolist()
            pred_mask_points.append(points)
        # 找到pred_mask_boxes
        pred_mask_boxes = []
        for pm in pred_mask_points:
            pred_mask_boxes.append(get_one_point_box(pm))

        # print(true_mask_boxes)
        # print(pred_mask_boxes)
        # print(true_mask_points)
        # print(pred_mask_points)
        has_check_boxes = set()
        for true_box in true_mask_boxes:
            for idx, (x, y) in enumerate(intersection_res):
                is_in = (true_box[1][0] <= x <= true_box[1][2]) and (true_box[1][1] <= y <= true_box[1][3])
                if is_in:
                    has_check_boxes.add(true_box)
                    break
        mistake_box = set(true_mask_boxes) - has_check_boxes
        # 找到轮廓内的误判,再从漏检中去掉
        in_circle = set()
        for box in pred_mask_boxes:
            for each_circle in true_mask_points:
                for idx, (x, y) in enumerate(each_circle):
                    temp = (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])
                    if temp and (idx != len(each_circle) - 1):
                        continue
                    elif temp and (idx == len(each_circle) - 1):
                        in_circle.add(get_one_point_box(each_circle))
                    else:
                        break
        final_mistake = set()
        for mb in mistake_box:
            if mb[1] in in_circle:
                continue
            else:
                final_mistake.add(mb)
        mistake_box = final_mistake
        # print(mistake_box)
        has_check_boxes = set(true_mask_boxes) - mistake_box
        has_check_nums = len(has_check_boxes)
        mistake_nums = len(mistake_box)
        print(file + ': 检测出的个数: ', has_check_nums)
        print(file + ': 漏检个数： ', mistake_nums)
        all_mistake_nums += mistake_nums
        # overkill_boxes = check_overkill(pred_mask_boxes, intersection_res)
        # print(file + ': 过杀个数：', len(overkill_boxes))

        # draw mistake boxes and has checked boxes in true img
        # if (mistake_nums > 0 and has_check_nums > 0) and mistake_nums / has_check_nums > 0.5:

        true_img = cv2.imread(img_path)
        for box in mistake_box:
            label, point = box[0], box[1]
            cv2.rectangle(true_img, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), (255, 0, 0), 8)
            left_down_location = (int(point[0]), int(point[3]))
            true_img = show_chinese(true_img, label, left_down_location)
        for box in has_check_boxes:
            label, point = box[0], box[1]
            cv2.rectangle(true_img, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), (0, 255, 0), 8)
            left_down_location = (int(point[0]), int(point[3]))
            true_img = show_chinese(true_img, label, left_down_location)
        # for point in overkill_boxes:
        #     cv2.rectangle(true_img, (point[0], point[1]), (point[2], point[3]), (0, 0, 255), 8)

        plt.figure(figsize=(14, 7), dpi=160)
        plt.subplot(1, 2, 1)
        plt.title('pred img')
        pred_img = cv2.imread(pred_img_path)
        plt.imshow(pred_img)

        plt.subplot(1, 2, 2)
        # plt.title(true_img + str(miou))
        plt.title('true img')
        plt.imshow(true_img)
        plt.show()
    print('avg ', all_mistake_nums / len(os.listdir(test_data_path)))


def get_parse():
    train_data = r'D:\py_program\testAll\segement\src\data\img/'
    train_label = r'D:\py_program\testAll\segement\src\data\mask/'
    test_data = r'D:\files\data\test_data/'
    test_label = r'D:\files\data\test_mask/'

    img_save_path = r'D:\files\data\save_img/'
    mask_save_path = r'D:\files\data\save_mask/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=320, help='model training epochs')
    parser.add_argument('--weight', type=str,
                        # default='./params/liteseg_shuffleNet_zdm_ep400_BCE_640x640_selfResize_best.pth',
                        default='./params/liteseg_mobileNet_zdm_ep320_BCE_640x640_selfResize_best.pth',
                        # default='./checkpoint/liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep120.pth',
                        help='weights path')
    parser.add_argument('--weight-last', type=str,
                        # default='./params/liteseg_shuffleNet_zdm_ep400_BCE_640x640_selfResize_last.pth',
                        default='./params/liteseg_mobileNet_zdm_ep320_BCE_640x640_selfResize_last.pth',
                        help='last epoch weights path')
    parser.add_argument('--train-data', type=str, default=train_data, help='train data path')
    parser.add_argument('--train-label', type=str, default=train_label, help='train label path')
    parser.add_argument('--test-data', type=str, default=test_data, help='test data path')
    parser.add_argument('--test-label', type=str, default=test_label, help='test label path')
    parser.add_argument('--img_save_path', type=str, default=img_save_path, help='to save pred img')
    parser.add_argument('--mask_save_path', type=str, default=mask_save_path, help='to save pred mask')

    parser.add_argument('--train_loss_curve_save_path', type=str, default='./train_loss_pic/',
                        help='train loss curve save path')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoint/liteseg_mobileNet_zdm_ep320_BCE_640x640_selfResize_best/',
                        help='checkpoint params save path')
    parser.add_argument('--go_on_epoch', type=int, default=100, help='checkpoint params epoch')
    parser.add_argument('--go_on_param', type=str,
                        default='liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep300.pth',
                        help='checkpoint go on params')

    parser.add_argument('--img-size', nargs='+', type=int, default=(640, 640), help='image size')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--input_channel', type=int, default=5, help='model input channels')
    parser.add_argument('--output_channel', type=int, default=1, help='model output channels')

    parser.add_argument('--threshold', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--class_number', type=int, default=2, help='segement label class number')
    parser.add_argument('--backbone', type=str, default='mobilenet', help='net backbone[mobile net/shuffle net]')

    opt = parser.parse_args()
    # print(opt)
    return opt


if __name__ == '__main__':
    draw_mistake()
