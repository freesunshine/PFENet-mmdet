# PFENet-mmdet
PFENet Code
1.	The code is based on mmdetection at https://github.com/open-mmlab/mmdetection. Documents provided by mmdetection can guide to understand and run our code.  
2.	PFENet base on resnet are implemented in "ppcn-mmdet\mmdet\models\backbones\polnet.py"
3.	Our dataset is in PASCAL VOC format. Data IO files are under "ppcn-mmdet\mmdet\datasets".  
  "pol_xml.py" is for raw polarization images(I0, I45, I90, I135) which stored in a 4-channel .tiff file ;   
  "npz_xml" is for traditional polarization parameter images such as DoLP, AoP and S0 which stored in .npz file;  
  "bgr_xml" is for RGB images.  
4.	mmdetection config files for PFENet is in “\configs\PPCN”
