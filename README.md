# ModelFoundry

A Tool for DNN Modularization to Support On-demand Model Reuse

MORE RESULTS can be seen in :

```bash
/results/README.md
```

## Abstract

DNN model reuse is a popular way to improve the efficiency of model construction, particularly with the massive models available on sharing platforms (e.g., HuggingFace). Recently, on-demand model reuse has drawn much attention, which aims to reduce the overhead and safety risk of model reuse via decomposing models into modules and reusing modules according to user's requirements. However, existing efforts for on-demand model reuse stop at algorithm implementations. These implementations involve ad-hoc decomposition in experiments and require considerable manual effort to adapt to new models; thus obstructing the practicality of on-demand model reuse.  In this paper, we introduceModelFoundry, a tool that systematically integrates two modularization approaches proposed in our prior work.  ModelFoundry provides automated and scalable model decomposition and module reuse functionalities, making it more practical and easily integrated into model sharing platforms. Evaluations conducted on widely used models sourced from PyTorch and GitHub platforms demonstrate thatModelFoundry achieves effective model decomposition and module reuse, as well as scalability to various models.

## Requirements

**This tool should be deployed on Linux with Nvidia GPU**

**For WebUI:**

- node v16.20.2
- npm v8.19.4
- vue v2.6.11

**For server:**

- Python v3.9.18
- Pytorch v1.8.1+cu111
- Torchvision v0.9.1+cu111
- Torchaudio v0.8.1
- Argparse v1.4.0
- Flask v3.0.0
- Werkzeug v3.0.0
- GPU with CUDA support

## Structure of the directories

```
[todo]
```

## How to use
Launch from docker image or source code!

### For Docker Launch

Docker images can be found in:

https://hub.docker.com/repository/docker/bxh1/modelfoundry

To get the docker images, run:

```bash
docker pull bxh1/modelfoundry:frontend_V2.1
docker pull bxh1/modelfoundry:backend_V2.1
```
**IMPORTANT:** mount the data in docker-compose.yml volumes:

```yaml
# Mount your data from "https://mega.nz/file/tX91ACpR#CSbQ2Xariha7_HLavE_6pKg4FoO5axOPemlv5J0JYwY" to /app/GradSplitter_main/data
    - /data/bixh/ToolDemo_GS/GradSplitter_main/data:/app/GradSplitter_main/data
# Mount your data from "https://mega.nz/folder/ADMjESyC#LkCOzE0qVHs8DOXkN3l_WA" to /app/SeaM_main/data
    - /data/bixh/ToolDemo_GS/SeaM_main/data:/app/SeaM_main/data
# If downloading trained models takes much time, please mount it to: /root/.cache/torch/hub/checkpoints
    - /data/bixh/ToolDemo_GS/checkpoints:/root/.cache/torch/hub/checkpoints
# And please mount your Imagenet dataset to /app/SeaM_main/data/dataset
    - /data/qibh/others/ILSVRC2012:/app/SeaM_main/ILSVRC2012
```

**Docker compose file can be found in this Repo**

To launch docker images and save logs, please modify the image name in docker-compose.yml, and use:

```bas
docker-compose up >logs.txt
```

### Launch from source code

```bash
git clone https://github.com/1836533846/ModelFoundry
cd ModelFoundry
```

The tool is divided into two parts.

###  For WebUI:

```bash
cd vue_project
```

To install requirements, please run:

```bash
npm install
```

And for running the web UI:

```bash
npm run serve
```

### For server:

```bash
cd flask_project
```

Python 3.9 and GPU with CUDA is required.

To install requirements, run

```ba
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

And start the server:

```ba
flask run
```

## Data

Move [Data](https://mega.nz/file/tX91ACpR#CSbQ2Xariha7_HLavE_6pKg4FoO5axOPemlv5J0JYwY) to:

```bash
ModelFoundry/flask_project/GradSplitter_main/data
```

Move [Data](https://mega.nz/folder/ADMjESyC#LkCOzE0qVHs8DOXkN3l_WA) to:

```bash
ModelFoundry/flask_project/SeaM_main/data
```

If functions related to ImageNet is required, please remove ImageNet dataset to 

```bash
ModelFoundry/flask_project/SeaM_main/ILSVRC2012
```
## Architecture

![workflow](./img/workflow.jpg)

## UI design

As shown in Figure below, the user interface of ModelFoundry is designed to be simple and intuitive. A user first searches for the required module and confirms which dataset the module comes from by viewing the module card. Then, the user specifies the model that needs to be decomposed. \projectName will directly evaluate the module and make download function available when the module exists. Next, the user can select the target task and click the **Modularize** button to send the task configurations to the server to execute the corresponding pipeline. The results will be sent back to the log box of the web interface, showing the evaluation of the decomposition. The decomposed modules can be downloaded by clicking **download**, or by clicking **reuse** to perform the reuse function we offer. 

![ModelFoundry](./img/ModelFoundry.jpg)
## Results
| **TC** | **CIFAR-10 (Acc./Wrr.)** | **CIFAR-10 (Acc./Wrr.)** | **CIFAR-10 (Acc./Wrr.)** | **SVHN (Acc./Wrr.)** | **SVHN (Acc./Wrr.)** | **SVHN (Acc./Wrr.)** |
|:------:|:------------------------:|:------------------------:|:------------------------:|:-------------------:|:-------------------:|:-------------------:|
|        | **InceCNN**              | **SimCNN**               | **ResCNN**               | **InceCNN**         | **SimCNN**          | **ResCNN**          |
| 0      | 95.75/9.40               | 95.65/5.40               | 95.95/8.79               | 97.74/5.77          | 97.88/4.29          | 97.76/5.53          |
| 1      | 98.40/5.20               | 98.30/3.47               | 98.50/5.57               | 97.63/4.97          | 97.90/3.48          | 97.75/4.10          |
| 2      | 93.70/10.79              | 91.35/6.99               | 91.70/11.29              | 97.43/5.26          | 97.67/3.60          | 97.41/4.27          |
| 3      | 90.80/14.10              | 88.65/7.49               | 90.75/11.92              | 95.33/6.30          | 95.35/3.99          | 95.26/5.69          |
| 4      | 94.80/11.18              | 95.25/5.15               | 95.60/8.50               | 98.08/5.10          | 98.14/3.59          | 98.08/4.56          |
| 5      | 92.70/12.17              | 92.55/5.97               | 93.65/9.86               | 96.96/5.92          | 97.76/4.06          | 96.96/5.50          |
| 6      | 97.00/7.34               | 96.80/5.29               | 97.05/8.48               | 96.66/6.87          | 97.17/4.77          | 96.89/6.39          |
| 7      | 96.45/9.66               | 95.90/5.55               | 96.25/8.59               | 97.37/6.29          | 97.97/4.58          | 97.72/5.69          |
| 8      | 97.40/6.95               | 96.65/4.67               | 96.85/7.06               | 96.39/7.47          | 96.78/5.16          | 95.81/6.79          |
| 9      | 97.70/6.50               | 97.55/4.39               | 97.40/6.89               | 97.05/7.20          | 96.83/5.02          | 97.37/6.60          |
| **Avg.**| **95.47/9.33**           | **94.86/5.44**           | **95.37/8.70**           | **97.06/6.12**      | **97.35/4.25**      | **97.10/5.51**      |

*The model decomposing results of \projectName with SeaM. "Acc." and "Wrr." denote the accuracy and weight retention rate of each module.*

| **Dataset** | **Model Name** | **Best TM Acc. (%)** | **Composed Model Acc. (%)** | **Improvement** |
|:-----------:|:--------------:|:--------------------:|:---------------------------:|:---------------:|
|             |                |                      |                             |                 |
| **CIFAR-10**|                |                      |                             |                 |
|             | SimCNN         | 81.01                | 86.26                       | 5.25            |
|             | ResCNN         | 81.88                | 85.95                       | 4.07            |
|             | InceCNN        | 83.06                | 86.94                       | 3.88            |
| **SVHN**    |                |                      |                             |                 |
|             | SimCNN         | 87.51                | 93.12                       | 5.61            |
|             | ResCNN         | 85.07                | 90.55                       | 5.48            |
|             | InceCNN        | 83.19                | 90.22                       | 7.03            |
| **Average** |                | -                    | -                           | 5.22            |

*The model reuse results of \projectName regarding composing more accurate model. "TM" denotes target model.*

