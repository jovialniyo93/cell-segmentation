# RA-SE-ASPP-Net
![Python](https://img.shields.io/badge/Python-V3.8-blue)
![Pytorch](https://img.shields.io/badge/Pytorch-V1.6-orange)
![CV2](https://img.shields.io/badge/CV2-V4.8-brightgreen)
![Pandas](https://img.shields.io/badge/Pandas-V1.4.2-ff69b4)
![Numpy](https://img.shields.io/badge/%E2%80%8ENumpy-V1.20.2-success)
![Releasedate](https://img.shields.io/badge/Release%20date-August2023-red)
![Opensource](https://img.shields.io/badge/OpenSource-Yes!-6f42c1)

# [Attention-Guided Residual U-Net with SE Connection and ASPP for Watershed-Based Cell Segmentation in Microscopy Images](https://doi.org/10.1089/cmb.2023.0446)

The code in this repository is supplementary to publication **Attention-Guided Residual U-Net with SE Connection and ASPP for Watershed-Based Cell Segmentation in Microscopy Images**
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB
* This project is writen in Python 3 and makes use of tensorflow.
  
# How to train and test our model

To train the model run the script```python train.py```.

To test the model run the script```python evaluate.py```.
<br/>

**To run the scripts, You can alternatively use Google Colaboratory:**

```python
UNET_iPSField7_Final_Figures_of_unet_train.ipynb
```



## Hyperparameters:
 
 <ol>
  <li>Batch size = 16</li> 
  <li>Number of epoch = 100</li>
  <li>Loss = Binary crossentropy</li>
  <li>Optimizer = Adam</li>
  <li>Dropout_rate =  0.05</li>
  <li>Learning Rate = 1e-4 (Adjusted for some experiments)</li>
</ol>

## Evaluation and metrics

**We use the following evaluation Metrics for experimental results:**

Accuracy, Jaccard, Dice, Precision, and Recall


## Dataset

**To download Dataset and all procedures for data preparation you can use this link:** [Click Here](https://github.com/jovialniyo93/cell-detection-and-tracking)	


# Project Collaborators and Contact

**Created by:** Ph.D. student: Jovial Niyogisubizo 
Department of Computer Applied Technology,  
Center for High Performance Computing, Shenzhen Institute of Advanced Technology, CAS. 

For more information, contact:

* **Prof Yanjie Wei**  
Shenzhen Institute of Advanced Technology, CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
yj.wei@siat.ac.cn


* **Jovial Niyogisubizo**  
Shenzhen Institute of Advanced Tech., CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
jovialniyo93@gmail.com

If you find our work useful in your research, please consider citing:

Jovial Niyogisubizo, Keliang Zhao, Jintao Meng, Yi Pan, Rosiyadi Didi, and Yanjie Wei , 2024.Attention-Guided Residual U-Net with SE Connection and ASPP for Watershed-Based Cell Segmentation in Microscopy Images. Journal of Computational Biology. https://doi.org/10.1089/cmb.2023.0446

>[Online Published Paper](https://doi.org/10.1089/cmb.2023.0446)

## License ##
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
