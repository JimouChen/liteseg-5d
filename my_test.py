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
from utils import SegmentationMetric, split_raw, get_parse, paste_evaluation

warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()
args = get_parse()

model = liteseg.LiteSeg(num_class=1,
                        backbone_network=args.backbone,
                        # backbone_network='mobilenet',
                        pretrain_weight=None,
                        is_train=False)

model.load_state_dict(torch.load(args.weight))

# checkpoint_path = args.checkpoint_path + args.go_on_param
# checkpoints = torch.load(args.weight)
# model.load_state_dict(checkpoints['model_state_dict'])
# torch.save(model.state_dict(), './params/ep120_mobile_liteseg.pth')

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(args.img_size),
])


def five_channel_test():
    model.eval()
    if os.path.exists(args.img_save_path):
        shutil.rmtree(args.img_save_path)
    if os.path.exists(args.mask_save_path):
        shutil.rmtree(args.mask_save_path)

    os.mkdir(args.img_save_path)
    os.mkdir(args.mask_save_path)

    m_dice = 0
    m_pa = 0
    m_cpa = 0
    m_mpa = 0
    m_mIoU = 0
    i = 1
    num = len(os.listdir(args.test_data))
    for raw_file in os.listdir(args.test_data):
        p = args.test_data + raw_file
        img = io.imread(p)
        img = np.transpose(img, (1, 2, 0))
        all_sub_img, start_location = split_raw(img, overlap_size=64)
        pred_img = np.zeros(img.shape[:2])
        temp_cat_list = []
        pos_x_y = []
        for idx, box in enumerate(start_location):
            # x1, y1, x2, y2 = box
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
                # print(cur_test_img.shape)
                out = model(cur_test_img)
                for idx_, each in enumerate(out):
                    # 贴回去pred_img
                    sub = torch.reshape(each, args.img_size)
                    sub = sub.detach().numpy()
                    # sub = torch.reshape(each.cpu(), args.img_size)
                    # sub = sub.cpu().detach().numpy()
                    x1, y1, x2, y2 = pos_x_y[idx_]
                    pred_img[y1:y2, x1:x2] = sub
                    # 要注意并集的两者的shape和dtype都要一样
                    # pred_img[y1:y2, x1:x2] = cv2.bitwise_or(pred_img[y1:y2, x1:x2], sub)

                pos_x_y = []
                # out = torch.reshape(out.cpu(), args.img_size)
                # print(out.shape)
                # out = out.cpu().detach().numpy()
                # return
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

        # 模型评估
        metric = SegmentationMetric(args.class_number)
        imgPredict = np.array(pred_img.reshape(-1), np.uint8)
        imgLabel = cv2.imread(args.test_label + raw_file.split('.')[0] + '.png', 0)
        # imgLabel = cv2.resize(imgLabel, args.img_size)
        imgLabel = np.array((imgLabel / 255).reshape(-1), np.int8)

        metric.addBatch(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        dice = mc.binary.dc(imgPredict, imgLabel)

        print('\n\n', '**==**==' * 50)
        print(f'第{i}张测试图片')
        print('m_dice:', dice)
        print('像素准确率 PA is : %f' % pa)
        print('类别像素准确率 CPA is :', cpa)
        print('类别平均像素准确率 MPA is : %f' % mpa)
        print('mIoU is : %f' % mIoU, end='\n\n')

        save_path = os.path.join(args.img_save_path, 'test_img_' + raw_file)
        paste_evaluation(save_img, mIoU, save_path)
        # cv2.imwrite(os.path.join(args.img_save_path, 'test_img_' + raw_file), save_img)
        cv2.imwrite(os.path.join(args.mask_save_path, 'test_mask_' + raw_file.split('.')[0] + '.png'), mask_img_to_save)
        print(raw_file + '  save res ok')

        m_pa += pa
        m_cpa += cpa
        m_mpa += mpa
        m_mIoU += mIoU
        m_dice += dice
        i += 1

    print('\n\n', '**==**==' * 50)
    print('m_dice:', m_dice / num)
    print('all 像素准确率 AVG_PA is : %f' % (m_pa / num))
    print('all 类别像素准确率 AVG_CPA is :', m_cpa / num)
    print('all 类别平均像素准确率 AVG_MPA is : %f' % (m_mpa / num))
    print('AVG_mIoU is : %f' % (m_mIoU / num), end='\n\n')


if __name__ == '__main__':
    five_channel_test()
