# SAVI - Trabalho Prático 2

Sistemas Avançados de Visualização Industrial (SAVI) - Grupo 3 - Universidade de Aveiro - 2023/24

## Table of Contents

* [Introduction](#introduction)
* [Datasets Used](#datasets-used)
* [Libraries Used](#libraries-used)
* [Installation](#installation)
* [Code Explanation](#code-explanation)
* [Authors](#authors)


---
## Introduction

<p align="justify"> In this assignment, a point cloud based model was created and trained to guess objects displayed in different scenes. This program needs to pre-process the scene to retrieve each object and its properties to feed the model, narrating the object prediction and its characteristics through a text-to-speech library. Furthermore, this model was also applied to a real-time system using a RGB-D camera.</p>

---
## Datasets Used

To train the aforementioned classifier, it was used the [Washington RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/). Therefore, it was used:
- [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/) - this is a point cloud dataset of each object used to train the model;
- [RGB-D Scenes Dataset V2](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/) - this a point cloud dataset of each scene, where each object to test the model was retrieved.

---
## Libraries Used

To run the program and scripts presented in this repository, some libraries need to be installed beforehand. These are the following:

- **[Open3D](https://www.open3d.org/)**
  - <u>Description</u>: library that allows for rapid reading, manipulating and storing of 3D data, like point clouds.
  - <u>Installation</u>:
    ```bash
    pip install open3d
    ```

- **[Torch](https://pytorch.org/)**
  - <u>Description</u>: PyTorch, or just Torch, is a fully featured framework for building deep learning models, which is a type of machine learning that's commonly used in applications like image recognition and  processing.
  - <u>Installation</u>:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

- **[Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)**
  - <u>Description</u>: this is a convenient library that allows to compute the performance of deep learning models in an iterative fashion.
  - <u>Installation</u>:
    ```bash
    pip install torchmetrics
    ```
  
- **[Torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/)**
  - <u>Description</u>: library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
  - <u>Installation</u>:
    ```bash
    pip install torch_geometric
    ```

- **[gTTS](https://gtts.readthedocs.io/en/latest/)**
  - <u>Description</u>: Google Text-to-Speech (gTTS) is a library to interface with Google Translate's text-to-speech API that allows writing spoken mp3 data to a file, allowing for further audio manipulation.
  - <u>Installation</u>:
    ```bash
    pip install gTTS
    ```

<!-- Add more libraries as needed -->

---
## Installation

To ensure the program runs as intended, the steps presented below should be followed.

1. Clone the repository:
```bash
git clone https://github.com/Goncalo287/savi_t2/
```
2. Change into the project directory:
```bash
cd savi_t2
```
3. Run the program:
```bash
./main.py
```

---
## Code Explanation 

<details >
<summary>Training the model</summary>

To train the model with Pointclouds information, a [PointNet](http://stanford.edu/~rqi/pointnet/) architecture was utilized. It consumes an entire point cloud, learns a spatial encoding of each point, aggregates learned encodings into features, and feeds them into a classifier. One advantage of this architecture is that it learns the global representation of the input, ensuring that the results are independent of the orientation of the Pointcloud. In this network architecture, there are several shared MLPs (1D Convolutions) from which critical points are extracted using a Max Pooling function. These critical points (outputs) feed into a classifier that predicts each object class. Aditional detailed information about this architecture can be found at ["An Intuitive Introduction to Point Net"](https://medium.com/@itberrios6/introduction-to-point-net-d23f43aa87d2).

To optimize the classifier parameters, a PointNetLoss function was implemented. In this function, the [Negative Log Likelihood Loss (NLLLOSS)](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) criterion was used to refine the model parameters during training to improve validation results. To prevent overfitting during training, the model was only saved when the validation error was minimal compared to the training process.

</details>

<details >
<summary>Scene Pre processing</summary>

aaa

</details>


---
## Authors

These are the contributors who made this project possible:

- **[Adriano Figueiredo](https://github.com/AdrianoFF10)**
  - Information:
    - Email: adrianofigueiredo7@ua.pt
    - NMec: 104192

- **[Gonçalo Anacleto](https://github.com/Goncalo287)**
  - Information:
    - Email: goncalo.anacleto@ua.pt
    - NMec: 93394

- **[Ricardo Bastos](https://github.com/RBastos36)**
  - Information:
    - Email: r.bastos@ua.pt
    - NMec: 103983
