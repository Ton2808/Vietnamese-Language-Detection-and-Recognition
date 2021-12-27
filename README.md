##### Table of Content

1. [Introduction (Khôi viết)](#dictionary-guided-scene-text-recognition)
1. [Dataset (đổi link thui thành 3k5 ảnh mình)](#dataset)
1. [Getting Started (An Viết)](#getting-started)
	- [Requirements](#requirements)
	- [Usage Example](#usage)
1. [Training & Evaluation (Tấn + Quỳnh viết)](#training-and-evaluation)
1. [Acknowledgement (đổi link thui)](#acknowledgement)

# Dictionary-guided Scene Text Recognition

- We propose a novel dictionary-guided sense text recognition approach that could be used to improve many state-of-the-art models.


| ![architecture.png](https://user-images.githubusercontent.com/32253603/117981172-ebd78580-b35e-11eb-84fe-b97c8d15d8bf.png) |
|:--:|
| *Comparison between the traditional approach and our proposed approach.*|

Details of the dataset construction, model architecture, and experimental results can be found in [our following paper](https://www3.cs.stonybrook.edu/~minhhoai/papers/vintext_CVPR21.pdf):

```
@inproceedings{m_Nguyen-etal-CVPR21,
      author = {Nguyen Nguyen and Thu Nguyen and Vinh Tran and Triet Tran and Thanh Ngo and Thien Nguyen and Minh Hoai},
      title = {Dictionary-guided Scene Text Recognition},
      year = {2021},
      booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    }
```
**Please CITE** our paper whenever our dataset or model implementation is used to help produce published results or incorporated into other software.

---

### Dataset

We use a dataset based on [VinText dataset](https://www3.cs.stonybrook.edu/~minhhoai/papers/vintext_CVPR21.pdf) combine with different picture we found on google. 

|    Name  						  | #imgs | #text instances						   | Examples 									|
|:-------------------------------:|:-----:|:-----------------------------------|:----------------------------------:|
|VinTextV2| 3800  | About 92000 			   |![example.png](https://user-images.githubusercontent.com/32253603/120605880-c67afa80-c478-11eb-8a2a-039a1d316503.png)|


|    Dataset variant  						  | Input format |  Link download 									|
|:-------------------------------:|:-----:|:----------------------------------:|
|Original| x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT  |[Download here](https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing)|
|Converted dataset| [COCO format](https://cocodataset.org/#format-data)  |[Download here](https://drive.google.com/file/d/1AXl2iOTvLtMG8Lg2iU6qVta8VuWSXyns/view?usp=sharing)|
### VinTextV2
Extract data and copy folder to folder ```Vietnamese-Language-Detection-and-Recognition/datasets/```

```
datasets
└───vintext
	└───test.json
		│train.json
		|train_images
		|test_images
└───evaluation
	└───gt_vintext.zip
```
---
### Add your own dataset
You can download our label tool [here](https://drive.google.com/drive/folders/1XcUnjJ2eOcXM0JOlYjAVQnnKP2Xx8FHV?usp=sharing)
![tools](images/label_tool.png)

##### Requirements

- python=3.7
- torch==1.8.0

##### Installation

```sh
conda create -n fb -y python=3.7
conda activate fb

#for CPU users
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
#for GPU users
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

python -m pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
### Check out the code and install: 
```sh
git clone https://github.com/Ton2808/Vietnamese-Language-Detection-and-Recognition.git
cd dict-guided
python setup.py build develop
```

##### Download vintext pre-trained model

- [trained_model](https://drive.google.com/file/d/15rJsQCO1ewJe-EInN-V5dSCftew4vLRz/view?usp=sharing).

##### Usage

### Run on Ubuntu or Windows

Prepare folders
```sh
mkdir sample_input
mkdir sample_output
```
Copy your images to ```sample_input/```. Output images would result in ```sample_output/```
```sh
python demo/demo.py --config-file configs/BAText/VinText/attn_R_50.yaml --input data/test_data --output sample_output/ --opts MODEL.WEIGHTS ./save_models/bbox.pth
```
| ![qualitative results.png](https://user-images.githubusercontent.com/32253603/120606555-836d5700-c479-11eb-9a37-09fa8cc129f3.png) |
|:--:|
| *Qualitative Results on VinText.*|

### Run on Docker

Download and unzip FantasticBeasts.zip
- [FantasticBeasts.zip](https://drive.google.com/file/d/10qYIcp8HIukuwPMnvpoWN7wTmQgXzm7p/view?usp=sharing).

```sh
unzip FantasticBeasts.zip
cd FantasticBeasts/Docker
```

Open Docker and load docker images
```sh
#for CPU users
docker load fantasticbeasts-aic-cpu.tar

#for GPU users with NIVIDA docker toolkit
docker load fantasticbeasts-aic-gpu.tar
```

Create folders and copy the path
```sh
mkdir test_data
mkdir submission_output
```

Run Docker images
```sh
#for CPU users
docker run --mount type=bind,source=[test_data_folder_path],target=/home/ml/AIC/aicsolution/data/test_data --mount type=bind,source=[submission_output_folder_path],target=/home/ml/AIC/aicsolution/data/submission_output [IMAGE ID] /bin/bash run.sh

#for GPU users with NIVIDA docker toolkit
nvidia-docker run -it --rm --gpus all --mount type=bind,source=[test_data_folder_path],target=/home/ml/AIC/aicsolution/data/test_data --mount type=bind,source=[submission_output_folder_path],target=/home/ml/AIC/aicsolution/data/submission_output [image ID] /bin/bash run.sh
```

### Training and Evaluation


#### Training

For training, we employed the pre-trained model [tt_attn_R_50](https://cloudstor.aarnet.edu.au/plus/s/tYsnegjTs13MwwK/download) from the ABCNet repository for initialization.

```sh
python tools/train_net.py --config-file configs/BAText/VinText/attn_R_50.yaml MODEL.WEIGHTS path_to_tt_attn_R_50_checkpoint
```

Example:
```sh
python tools/train_net.py --config-file configs/BAText/VinText/attn_R_50.yaml MODEL.WEIGHTS ./tt_attn_R_50.pth
```

Trained model output will be saved in the folder ```output/batext/vintext/``` that is then used for evaluation

#### Evaluation

```sh
python tools/train_net.py --eval-only --config-file configs/BAText/VinText/attn_R_50.yaml MODEL.WEIGHTS path_to_trained_model_checkpoint
```
Example:
```sh
python tools/train_net.py --eval-only --config-file configs/BAText/VinText/attn_R_50.yaml MODEL.WEIGHTS ./output/batext/vintext/trained_model.pth
```
### Acknowledgement
This repository is built based-on [ABCNet](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText)
