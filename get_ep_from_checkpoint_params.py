import shutil
import warnings
import torch
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image
from skimage import io
from medpy import metric as mc
import gc

from LiteSeg import liteseg
from utils import *

warnings.filterwarnings('ignore')
args = get_parse()

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(args.img_size),
])


def five_channel_test(model):
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    if os.path.exists(args.img_save_path):
        shutil.rmtree(args.img_save_path)
    if os.path.exists(args.mask_save_path):
        shutil.rmtree(args.mask_save_path)

    os.mkdir(args.img_save_path)
    os.mkdir(args.mask_save_path)

    # m_dice = 0
    # m_pa = 0
    # m_cpa = 0
    # m_mpa = 0
    m_mIoU = 0
    i = 1
    num = len(os.listdir(args.test_data))
    for raw_file in os.listdir(args.test_data):
        cur_all_sub_out_iou = []
        p = args.test_data + raw_file
        p_mask = args.test_label + raw_file.split('.')[0] + '.png'
        raw_mask = cv2.imread(p_mask, 0)
        img = io.imread(p)
        img = np.transpose(img, (1, 2, 0))
        all_sub_img, start_location = split_raw(img, overlap_size=64)
        pred_img = np.zeros(img.shape[:2])
        temp_cat_list = []
        pos_x_y = []
        for idx, box in enumerate(start_location):
            # x11, y11, x22, y22 = box
            pos_x_y.append(box)
            img = np.array(all_sub_img[idx] / 255., np.float32)
            # img = np.array(all_sub_img[idx])
            test_img = transform(img)[None]
            test_img_tensor = torch.tensor(test_img)
            temp_cat_list.append(test_img_tensor)
            if (idx + 1) % 4 == 0:
                # print(pos_x_y)
                cur_test_img = torch.cat(temp_cat_list, 0)
                temp_cat_list = []
                # cur_test_img = cur_test_img.cuda()
                out = model(cur_test_img)
                for idx_, each in enumerate(out):
                    # 贴回去pred_img
                    sub = torch.reshape(each, args.img_size)
                    sub = sub.detach().numpy()
                    x1, y1, x2, y2 = pos_x_y[idx_]
                    pred_img[y1:y2, x1:x2] = sub
                    sub_out_miou = evaluation_model(sub, raw_mask[y1:y2, x1:x2], args)
                    cur_all_sub_out_iou.append(sub_out_miou)
                pos_x_y = []
                # out = torch.reshape(out.cpu(), args.img_size)
                # out = out.cpu().detach().numpy()
                # pred_img[y1:y2, x1:x2] = out
        pred_img = np.where(pred_img > args.threshold, 1, 0)
        # 把图片贴合回去展示
        # draw in mask
        save_img = Image.open(args.test_data + raw_file)
        # save_img = save_img.resize(args.img_size)
        mask_img_to_save = np.array(pred_img * 255, np.uint8)
        mask_img = cv2.cvtColor(mask_img_to_save, cv2.COLOR_GRAY2RGBA)

        h, w, c = mask_img.shape
        mask_img[:, :, 3] = 50
        mask = Image.fromarray(np.uint8(mask_img))
        b, g, r, a = mask.split()
        save_img.paste(mask, (0, 0, w, h), mask=a)
        save_img = np.array(save_img)
        # print(save_img)

        # cv2.imwrite(os.path.join(args.img_save_path, 'test_img_' + raw_file), save_img)
        # cv2.imwrite(os.path.join(args.mask_save_path, 'test_mask_' + raw_file.split('.')[0] + '.png'), mask_img_to_save)
        # print(raw_file + '  save res ok')
        # 模型评估
        # metric = SegmentationMetric(args.class_number)
        # imgPredict = np.array(pred_img.reshape(-1), np.uint8)
        # imgLabel = cv2.imread(args.test_label + raw_file.split('.')[0] + '.png', 0)
        # imgLabel = cv2.resize(imgLabel, args.img_size)
        # imgLabel = np.array((imgLabel / 255).reshape(-1), np.int8)

        # metric.addBatch(imgPredict, imgLabel)
        # pa = metric.pixelAccuracy()
        # cpa = metric.classPixelAccuracy()
        # mpa = metric.meanPixelAccuracy()
        # mIoU = metric.meanIntersectionOverUnion()
        # dice = mc.binary.dc(imgPredict, imgLabel)

        # print('\n\n', '**==**==' * 50)
        # print(f'第{i}张测试图片')
        # print('m_dice:', dice)
        # print('像素准确率 PA is : %f' % pa)
        # print('类别像素准确率 CPA is :', cpa)
        # print('类别平均像素准确率 MPA is : %f' % mpa)
        # print('mIoU is : %f' % mIoU, end='\n\n')
        cur_avg_iou = sum(cur_all_sub_out_iou) / len(cur_all_sub_out_iou)
        save_path = os.path.join(args.img_save_path, 'test_img_' + raw_file)
        paste_evaluation(save_img, cur_avg_iou, save_path)
        # cv2.imwrite(os.path.join(args.img_save_path, 'test_img_' + raw_file), save_img)
        cv2.imwrite(os.path.join(args.mask_save_path, 'test_mask_' + raw_file.split('.')[0] + '.png'), mask_img_to_save)
        # print(raw_file + '  save res ok')

        # m_pa += pa
        # m_cpa += cpa
        # m_mpa += mpa
        # m_mIoU += mIoU
        # m_dice += dice

        # cur_avg_iou = m_mIoU / i
        print(raw_file, ' ==== cur avg iou: ', cur_avg_iou)
        m_mIoU += cur_avg_iou
        i += 1

    # print('\n\n', '**==**==' * 50)
    # print('m_dice:', m_dice / num)
    # print('all 像素准确率 AVG_PA is : %f' % (m_pa / num))
    # print('all 类别像素准确率 AVG_CPA is :', m_cpa / num)
    # print('all 类别平均像素准确率 AVG_MPA is : %f' % (m_mpa / num))
    # print('AVG_mIoU is : %f' % (m_mIoU / num), end='\n\n')

    return m_mIoU / num


def get_best_ep():
    checkpoint_path = args.checkpoint_path
    all_test_res = []
    for param in os.listdir(checkpoint_path):
        if '320' not in param:
            continue
        param_path = checkpoint_path + param
        checkpoints = torch.load(param_path)
        print('load... ', param_path)
        model = liteseg.LiteSeg(num_class=1,
                                # backbone_network='mobilenet',
                                backbone_network=args.backbone,
                                pretrain_weight=None,
                                is_train=False)
        model.load_state_dict(checkpoints['model_state_dict'])
        cur_avg_miou = five_channel_test(model)

        print(f'{param} avg mIOU: {cur_avg_miou}')
        all_test_res.append([param, cur_avg_miou])
    all_test_res.sort(key=lambda x: x[1], reverse=True)
    print(all_test_res)
    for r in all_test_res:
        print(r)


if __name__ == '__main__':
    get_best_ep()

'''res
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep120.pth', 0.6237571775497627]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep280.pth', 0.6220129565653076]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep260.pth', 0.6212794478659927]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep200.pth', 0.6168901561809532]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep300.pth', 0.6145839979073191]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep160.pth', 0.6140060737718238]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep180.pth', 0.6029063451686865]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep140.pth', 0.6011251339522588]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep100.pth', 0.5997486310162502]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep220.pth', 0.5988662106318707]
['liteseg_zdm_ep400_BCE_640x640_selfResize_best_ep240.pth', 0.598625672927424]
'''
'''
main pic:
shuffle net liteseg:
ep100 :0.6162654503623831
ep120 :0.6154555830948208
ep160 :0.5661063684433846
ep180 :0.6151603650986458
ep200 :0.6130643847404087
ep220 :0.6179767938430025  **
ep260 :0.6095356555025329
ep280 :0.616349402714124 *
ep300 :0.581963395541494
ep320 :0.6136783326154513
ep340 :0.610648511100478
ep360 :0.6056608747303377
ep380 :0.5963703244164091

==============================

new miou avg
['shuffleNet_LiteSeg_ep100.pth', 0.8044663915977597]
['shuffleNet_LiteSeg_ep120.pth', 0.8031655527817603]
['shuffleNet_LiteSeg_ep320.pth', 0.7962302747263739]
['shuffleNet_LiteSeg_ep260.pth', 0.7886579907448213]
['shuffleNet_LiteSeg_ep340.pth', 0.788201025641974]
['shuffleNet_LiteSeg_ep220.pth', 0.7855989322789694]
['shuffleNet_LiteSeg_ep360.pth', 0.7844494165811492]
['shuffleNet_LiteSeg_ep280.pth', 0.7839344514337258]
['shuffleNet_LiteSeg_ep380.pth', 0.7522334812689105]
['shuffleNet_LiteSeg_ep300.pth', 0.6950403242254068]
['shuffleNet_LiteSeg_ep160.pth', 0.6444681336424019]
'''
