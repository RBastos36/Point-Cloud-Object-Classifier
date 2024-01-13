# SAVI - Trabalho Prático 2

Sistemas Avançados de Visualização Industrial (SAVI) - Grupo 3 - Universidade de Aveiro - 2023/24

## Table of Contents

* [Introduction](#introduction)
* [Libraries Used](#libraries-used)
* [Datasets Used](#libraries-used)
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

Provide step-by-step instructions on how to install and set up your project. Include any prerequisites that users need to have installed.

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo.git](https://github.com/Goncalo287/savi_t2/)

# Change into the project directory
cd savi_t2

# Additional setup steps, if any
```

---
## Code Explanation 

Provide a concise explanation of your project.
Provide a concise explanation of your project.
Provide a concise explanation of your project.

```python
def add_numbers(a, b):
    return a + b

print("Hello, World!")
```

---
## Authors

These are the contributors who made this project possible:

- **[Adriano Figueiredo](https://github.com/AdrianoFF10)**
  - Informação:
    - Email: adrianofigueiredo7@ua.pt
    - Número Mecanográfico: 104192

- **[Gonçalo Anacleto](https://github.com/Goncalo287)**
  - Informação:
    - Email: goncalo.anacleto@ua.pt
    - Número Mecanográfico: 93394

- **[Ricardo Bastos](https://github.com/RBastos36)**
  - Informação:
    - Email: r.bastos@ua.pt
    - Número Mecanográfico: 103983
