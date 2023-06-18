# RA-SE-ASPP-Net

The code in this repository is supplementary to our future publication "Attention-Guided Residual U-Net with SE Connection and ASPP for Watershed-based Cell Segmentation in Microscopy Images" 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB
* This project is writen in Python 3 and makes use of tensorflow. 

## Installation
In order to get the code, either clone the project, or download a zip file from GitHub.

Clone the Cell segmentation repository:
```
https://github.com/jovialniyo93/cell-segmentation.git
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the Cell segmentation repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
```
```
conda env create -f requirements.yml
```
Activate the virtual environment cell-segmentation_ve:
```
conda activate cell-segmentation_ve
```

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

**The Dataset is available in the link below :**

[To download Dataset and all procedures for data preparation you can use this link:] [Click Here](https://github.com/jovialniyo93/cell-detection-and-tracking)	


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

## License ##
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
