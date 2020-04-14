from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import cv2
import sys

import numpy as np
import torch
from sklearn.preprocessing import normalize, MinMaxScaler


def load_pol_sub_image(sample_file, div_num=65535.0):
    img = cv2.imread(sample_file, -1)
    if img is None:
        print('load image error')
        print(sample_file)
    else:
        img = img.astype(np.float32)
        img = img / div_num
    return img


def fusion_one_image_one_ckpoint(img_path, cfg_path, pth_path, out_dir, method=0):
    sample = load_pol_sub_image(img_path)
    sample = sample.swapaxes(0, 2).swapaxes(1, 2)
    sample = sample.astype(np.float)
    sample = np.expand_dims(sample, axis=0)
    sample = torch.from_numpy(sample).float().cuda()

    model = init_detector(cfg_path, pth_path, device='cuda:0')
    result = model.backbone.fusion(sample).cpu()
    result = result.detach().numpy()[0, :]

    out_filename = img_path.split('/')[-1].split('.')[0]

    for i in range(result.shape[0]):
        out_path = os.path.join(out_dir, out_filename+"_"+str(i)+'.jpg')
        array_to_jpg(result[i, :], out_path, method)


def fusion_images_with_epocks(sample_dir, cfg_path, pth_dir, out_dir, method=0):
    sample_names = [i for i in os.listdir(sample_dir) if i.endswith('.tiff')]
    sample_paths = [os.path.join(sample_dir, i) for i in sample_names]

    pth_names = [i for i in os.listdir(pth_dir) if i.endswith('.pth') and i.startswith('epoch')]
    pth_paths = [os.path.join(pth_dir, i) for i in pth_names]

    for pth_name in pth_names:
        pth_path = os.path.join(pth_dir, pth_name)
        model = init_detector(cfg_path, pth_path, device='cuda:0')

        for sample_name in sample_names:
            sample_path = os.path.join(sample_dir, sample_name)
            sample = load_pol_sub_image(sample_path)
            sample = sample.swapaxes(0, 2).swapaxes(1, 2)
            sample = sample.astype(np.float)
            sample = np.expand_dims(sample, axis=0)
            sample = torch.from_numpy(sample).float().cuda()
            result = model.backbone.fusion(sample).cpu()
            result = result.detach().numpy()[0, :]

            out_filename = sample_name.split('.')[0] + "_" + pth_name.split('.')[0]
            for i in range(result.shape[0]):
                out_path = os.path.join(out_dir, out_filename + "_" + str(i) + '.jpg')
                array_to_jpg(result[i, :], out_path, method)


def array_to_jpg(arr, out_path, method=0):
    min_max_scaler = MinMaxScaler()
    if method == 0:
        result = min_max_scaler.fit_transform(arr)
    else:
        result = min_max_scaler.fit_transform(np.abs(arr))
    result = result * 255
    cv2.imwrite(out_path, result)
    print(out_path)


if __name__ == '__main__':
    # model = init_detector("/home/gdgc0402/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-5.py",
    #                       "/home/gdgc0402/Data/work_dirs/car-xmls/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-5/epoch_200.pth",
    #                       device='cuda:0')
    # print(model.backbone.fusion)

    # polnet_cfg = "/home/gdgc0402/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-5.py"
    # polnet_pth = "/home/gdgc0402/Data/work_dirs/car-xmls/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-5/epoch_200.pth"
    # polnet_sample = "/home/gdgc0402/Data/PolData/images/d04590135_images/20200102_102624628.tiff"
    # fusion_one_image_one_ckpoint(polnet_sample, polnet_cfg, polnet_pth, out_dir="/home/gdgc0402/")

    polnet_cfg = "/home/wangyong/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-9.py"
    pth_dir = "/home/wangyong/data/123"
    polnet_sample = "/home/wangyong/data/123/input"
    out_dir = "/home/wangyong/data/123/out/"
    fusion_images_with_epocks(polnet_sample, polnet_cfg, pth_dir, out_dir)
