# Unlocking the Power of SAM 2 for Few-Shot Segmentation

This repository contains the code for CVPR'25 submission #2456 titled "*Unlocking the Power of SAM 2 for Few-Shot Segmentation*".

> **Abstract**: *Few-Shot Segmentation (FSS) aims to learn class-agnostic segmentation on few base classes to segment arbitrary novel classes, but at the risk of overfitting. To address this, some methods use the well-learned knowledge of foundation models (e.g., SAM) to simplify the learning process. Recently, SAM 2 has extended SAM by supporting video segmentation, whose class-agnostic matching ability is useful to FSS. A simple idea is to encode support foreground (FG) features as memory, with which query FG features are matched and fused. Unfortunately, the FG objects in different frames of SAM 2's video data are always the same identity, while those in FSS are different identities, i.e., the matching step is incompatible. Therefore, we design Pseudo Prompt Generator to encode pseudo query memory, matching with query features in a compatible way. However, the memories can never be as accurate as the real ones, i.e., they are likely to contain incomplete query FG, but some unexpected query background (BG) features, leading to wrong segmentation. Hence, we further design Iterative Memory Refinement to fuse more query FG features into the memory, and devise a Support-Calibrated Memory Attention to suppress the unexpected query BG features in memory during matching. Extensive experiments have been conducted on PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> to validate the effectiveness of our design, e.g., the 1-shot mIoU can be 4.2% better than the best baseline. The code will be released upon paper acceptance.*

## Dependencies

- Python 3.12.7
- PyTorch 2.5.1
- cuda 12.1
- cudnn 9.1.0
```
> conda env create -f env.yaml
```

## Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

You can download the pre-processed PASCAL-5<sup>i</sup> dataset [here](https://drive.google.com/file/d/1Pt6ZDVcOw7iDzU2sP-Rxz_Us-lGx3QLP/view?usp=sharing) (COCO-20<sup>i</sup> is quite large, we will provide it upon paper acceptance), and extract it into `data/` folder. Then, you need to create a symbolic link to the `pascal/VOCdevkit` data folder as follows:
```
> ln -s <absolute_path>/data/pascal/VOCdevkit <absolute_path>/data/VOCdevkit2012
```

The directory structure is:

    ../
    ├── FSSAM/
    └── data/
        ├── VOCdevkit2012/
        │   └── VOC2012/
        │       ├── JPEGImages/
        │       ├── ...
        │       └── SegmentationClassAug/
        └── MSCOCO2014/           
            ├── annotations/
            │   ├── train2014/ 
            │   └── val2014/
            ├── train2014/
            └── val2014/

## Models

- Download the trained SAM 2 models (`sam2_hiera_small`, `sam2_hiera_base_plus`, `sam2_hiera_large`) from [the official SAM 2 repository](https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints) and put them into the `pretrained/` directory.
- Download [exp.tar.gz](https://drive.google.com/file/d/115UqM0cMGZtBqlIEb5Ma0aS2fkcDP8Sl/view?usp=sharing) to obtain trained FSSAM for PASCAL-5<sup>i</sup>. Note that we only provide 1-shot models trained on PASCAL-5<sup>i</sup>, and will provide other models upon paper acceptance.

## Commands

- **Training**:
    ```
    <sh train.sh {GPU: 4} {Port: 1234} {Dataset: pascal/coco} {Split: 0/1/2/3} {Shot: 1/5} {Model: FSSAM/FSSAM5s} {SAM 2: small/base/large}>

    # e.g., train split 0 under 1-shot setting on PASCAL-5<sup>i</sup>, with SAM 2 small:
    > sh train.sh 4 1234 pascal 0 1 FSSAM small

    # e.g., train split 0 under 5-shot setting on COCO-20<sup>i</sup>, with SAM 2 small:
    > sh train.sh 4 1234 coco 0 5 FSSAM5s small
    ```
- **Testing**:
    ```
    <sh test.sh {Dataset: pascal/coco} {Split: 0/1/2/3} {Shot: 1/5} {Model: FSSAM/FSSAM5s} {SAM 2: small/base/large}>

    # e.g., test split 0 under 1-shot setting on PASCAL-5<sup>i</sup>, with SAM 2 small:
    > sh test.sh pascal 0 1 FSSAM small

    # e.g., test split 0 under 5-shot setting on COCO-20<sup>i</sup>, with SAM 2 small:
    > sh test.sh coco 0 5 FSSAM5s small
    ```

## References

This repo is mainly built based on [HMNet](https://github.com/Sam1224/HMNet) and [SAM 2](https://github.com/facebookresearch/sam2). Thanks for their great work!
