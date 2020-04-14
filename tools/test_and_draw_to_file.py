from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import cv2
import sys
from mmdet.datasets.pipelines.loading import LoadPolNPZImageFromFile
from mmdet.datasets.pipelines.loading import LoadPolSubImageFromFile
import numpy as np


def load_pol_sub_image(sample_file, div_num=65535.0):
    img = cv2.imread(sample_file, -1)
    if img is None:
        print('load image error')
        print(sample_file)
    else:
        img = img.astype(np.float32)
        img = img / div_num
    return img


def load_pol_npz_image(sample_file):
    img = np.load(sample_file)["arr_0"]
    if img is None:
        print('load image error')
        print(sample_file)
    return img


def test_and_draw_from_single_file(sample_file, ext_name, bgr_file, out_file, config_file, checkpoint_file, score_threhold):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    sample = None
    if ext_name == 'bgr':
        sample = mmcv.imread(sample_file)
    elif ext_name == 'tiff':
        sample = load_pol_sub_image(sample_file)
    else:
        sample = LoadPolNPZImageFromFile(sample_file)
    print(sample)
    result = inference_detector(model, sample)
    img = mmcv.imread(bgr_file)
    show_result(img, result, model.CLASSES, out_file=out_file, score_thr=score_threhold)


# ext_name= bgr\pol\sub\others
def test_and_draw_from_xmls(xml_dir, ext_name, sample_dir, bgr_dir, out_dir, config_file, checkpoint_file, score_threhold):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    xlms = os.listdir(xml_dir)
    sample_ids = [i.split('_')[0]+'_'+i.split('_')[1] for i in xlms if i.endswith('.xml')]

    for xlm_filename in xlms:
        sample_file=''
        sample_path=''
        sample = None
        sample_id = xlm_filename.split('.')[0]
        if ext_name=='bgr':
            sample_file = xlm_filename.split('.')[0] + '.tiff'
        elif ext_name =='pol' or ext_name=='sub':

            sample_file = sample_id.split('_')[0] + '_' + sample_id.split('_')[1] + '.' + 'tiff'
        else:
            sample_file = sample_id.split('_')[0] + '_' + sample_id.split('_')[1] + '.' + 'ext_name'+'.npz'

        sample_path = os.path.join(sample_dir, sample_file)
        if ext_name=='bgr':
            sample = mmcv.imread(sample_path)
        elif ext_name =='pol' or ext_name=='sub':
            sample = load_pol_sub_image(sample_path)
        else:
            sample = load_pol_npz_image(sample_path)

        img_path = os.path.join(bgr_dir, xlm_filename.split('.')[0] + '.tiff')
        img = mmcv.imread(img_path)

        result = inference_detector(model, sample)

        out_file = sample_id.split('_')[0] + '_' + sample_id.split('_')[1] + '.' + ext_name+'.jpg'
        out_path = os.path.join(out_dir, out_file)

        show_result(img, result, model.CLASSES, show=False,  out_file=out_path, score_thr=score_threhold)

        print(out_path)


if __name__ == '__main__':

    # polnet_cfg = "/home/gdgc0402/Code/mmdet-pol/configs/PolNet/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-5.py"
    # polnet_pth = "/home/gdgc0402/Data/work_dirs/car-xmls/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-5/epoch_80.pth"
    # polnet_sample = "/home/gdgc0402/Data/PolData/images/d04590135_images/20200102_102624628.tiff"
    # polnet_bgr = "/home/gdgc0402/Data/PolData/images/bgr_images/20200102_102624628.tiff"
    #
    # test_and_draw_from_single_file(polnet_sample,
    #                                'tiff',
    #                                polnet_bgr,
    #                                '/home/gdgc0402/1.jpg',
    #                                polnet_cfg,
    #                                polnet_pth,
    #                                0.5
    #                                )
    #
    # bgr_cfg = "/home/gdgc0402/Code/mmdet-pol/configs/PolNet/faster_rcnn_bgr_r50_fpn_1x.py"
    # bgr_pth = "/home/gdgc0402/Data/work_dirs/car-xmls/faster_rcnn_bgr_r50_fpn_1x/epoch_80.pth"
    # bgr_sample = "/home/gdgc0402/Data/PolData/images/bgr_images/20200102_102624628.tiff"
    # bgr_bgr = "/home/gdgc0402/Data/PolData/images/bgr_images/20200102_102624628.tiff"
    #
    # test_and_draw_from_single_file(bgr_sample,
    #                                'bgr',
    #                                bgr_bgr,
    #                                '/home/gdgc0402/1.jpg',
    #                                bgr_cfg,
    #                                bgr_pth,
    #                                0.5
    #                                )
    xml_dir = '/home/wangyong/data/poldata/test-xml'
    ext_name = 'pol'
    sample_dir = '/home/wangyong/data/poldata/d04590135_images'
    bgr_dir = '/home/wangyong/data/poldata/bgr_images'
    out_dir = '/home/wangyong/data/poldata/result_images'
    config_file = "/home/wangyong/Code/mmdet-pol/configs/PPCN/faster_rcnn_pol_r50_fpn_1x_48-96-32-16-9.py"
    checkpoint_file = "/home/wangyong/Code/mmdet-pol/work_dirs/person_car2_faster_rcnn_polnet_r50_fpn_1x_48-96-32-16-9/epoch_80.pth"
    score_threhold = 0.5
    test_and_draw_from_xmls(xml_dir, ext_name, sample_dir, bgr_dir, out_dir, config_file, checkpoint_file, score_threhold)
