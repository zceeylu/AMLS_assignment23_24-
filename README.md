# AMLS_assignment23_24-SN20007754 (MLS project)
This repository is used to document the progress made for the AMLS assignment. 

## Project Overview


## Project Brief
The project consists of two tasks:
1. Task A - Binary classification task (using PneumoniaMNIST dataset). The objective is to classi-fy an image onto "Normal" (no pneumonia) or "Pneumonia" (presence of pneumonia)
2. Task B - Multi-class classification task (using PathMNIST dataset): The objective is to classify an image onto 9 different types of tissues.


## Running the Project
To run the project, run the following command:
```
python3 main.py
```

## Project Structure
The repository is structured as follows:

```
AMLS_23_24_SN20007754

|__A
|  |__DataReshape.py #contains
|  |__function.py #contains 
|  |__KNN.py #contains KNN model
|  |__RandomForest.py #contains RandomForest Model
|  |__SVM.py #contains SVM model

|__B
|  |__function.py #contains 
|  |__CNN.py #contains CNN model

|__Datasets #empty folder

|__results_and_analysis
|  |__Task_A #contains results for Task A
|  |__Task B #contains results for Task B

|__main.py #main file to run tasks

|__README.md #

|__requirements.txt #requirements to be installed

```

## Requirements
To install the required packages, run the following command:
```
pip install -r requirements.txt
```

## Acknowledgements
The datasets used in this project are from the MedMNIST repository, and are cited as follows:
```
Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.
                            
Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.
```
The instructions and inspiration for this project can be found in the ELEC0134 module, led by Dr. Miguel Rodrigues at UCL.
