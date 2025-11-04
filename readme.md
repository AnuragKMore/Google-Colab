MULTI-PATH SIAMESE CONVOLUTION NETWORK (MASCN)

OVERVIEW
This project implements the Multi-Path Siamese Convolution Network (MASCN), a deep learning framework for Offline Handwritten Signature Verification. The model aims to achieve high accuracy while maintaining a low model capacity, making it ideal for lightweight or mobile biometric verification systems.

The architecture is based on the research paper:
Xian Zhang et al. (2022), “Multi-Path Siamese Convolution Network for Offline Handwritten Signature Verification,” ICCDE 2022, Bangkok, Thailand.
DOI: 10.1145/3512850.3512854

CORE IDEA
MASCN is designed to learn discriminative features of handwritten signatures while minimizing background noise. It consists of three main components:

Multi-Scale Input Generation

Generates three scaled versions of each signature image (e.g., 50x150, 60x180, 70x210).

Maintains uniform stroke characteristics while varying background information.

Feature Extraction with Fusion Attention

A Siamese network with three weight-sharing branches.

Each branch uses Dense and Transition modules.

A Fusion Attention mechanism helps the model focus on stroke details.

Weighted Verification Module

Computes Euclidean distances for feature pairs from each branch.

Combines them using weighted coefficients (α=0.5, β=0.3, γ=0.2).

Uses Multi_Contrastive_Loss (MCL) to stabilize training.

MODEL DETAILS
Base Network: Siamese CNN with Dense and Transition modules
Attention Mechanism: Fusion attention to enhance stroke focus
Loss Function: Multi_Contrastive_Loss (MCL)
Distance Metric: Weighted Euclidean Distance
Framework: PyTorch
Backbone: Custom CNN inspired by DenseNet connections

DATASETS
MASCN is evaluated using both academic and publicly available datasets:

CEDAR Dataset – 55 users, 24 genuine and 24 forged signatures each.

BHSig260-Bengali – 100 users, 24 genuine and 30 forged signatures each.

BHSig260-Hindi – 160 users, 24 genuine and 30 forged signatures each.

Kaggle Handwritten Signatures – Public dataset by Divyansh Rai used for testing and additional evaluation.
Link: https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures

All datasets were used in writer-independent (WI) mode, with a 7:2:1 split for training, validation, and testing.

SETUP AND INSTALLATION
Requirements:

Python 3.8 or higher

PyTorch 1.10 or higher

CUDA (optional for GPU acceleration)

Install dependencies:
pip install torch torchvision numpy opencv-python tqdm matplotlib

Clone the repository:
git clone https://github.com/AnuragKMore/Google-Colab.git
cd MASCN

USAGE
Train the model:
python train.py --dataset cedar --epochs 50 --batch-size 16

Evaluate the model:
python test.py --dataset bengali --weights path_to_trained_model.pth

Verify signatures (single pair):
python verify.py --img1 path/to/genuine.png --img2 path/to/test.png

RESULTS SUMMARY
Dataset | Accuracy (%) | FAR | FRR | EER
CEDAR | 80.75 | 19.21 | 18.35 | 18.92
Bengali | 92.86 | 9.96 | 5.85 | 8.18
Hindi | 94.99 | 5.73 | 4.86 | 5.32
Kaggle | Empirical qualitative testing | — | — | —

MASCN achieves strong generalization and reduces the number of parameters compared to models like SigNet and IDN, making it practical for limited-resource environments.

REPOSITORY STRUCTURE
MASCN/
├── MASCN_extracted_1.py (core model)
├── db.py (test data)
├── link to the paper.txt (Research paper for the model)
└── README.txt (documentation)

CITATION
If you use this repository or model, cite the following:

@inproceedings{zhang2022mascn,
title={Multi-Path Siamese Convolution Network for Offline Handwritten Signature Verification},
author={Xian Zhang and Zhongcheng Wu and Liyang Xie and Yong Li and Fang Li and Jun Zhang},
booktitle={Proceedings of the 8th International Conference on Computing and Data Engineering (ICCDE 2022)},
year={2022},
doi={10.1145/3512850.3512854}
}

If you use the Kaggle dataset, please cite:
@misc{rai2021handwritten,
author = {Divyansh Rai},
title = {Handwritten Signatures},
year = {2021},
howpublished = {Kaggle},
url = {https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures}

}

