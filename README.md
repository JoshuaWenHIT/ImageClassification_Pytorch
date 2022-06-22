# <div align="center">ImageClassification_Pytorch</div>
<div align="center">Pytorch version of image classification implementation framework, which can apply standard datasets and custom datasets.</div>

## <div align="center">What is this repository ?</div>

**Firstly, what I have to say is: 
my framework was built on [this repository](https://github.com/hysts/pytorch_image_classification), 
and the original author has done a very good job. 
On this basis, I added the functions of training custom datasets and adapting custom networks, 
and added the test code and visualized results to fix the bugs of the original evaluation code.**

<center class = "half">
<img src = figures/confusion_matrix.png  height = 300><img src = figures/roc_curve.png  height = 300 >
</center>

## <div align="center">How to install this repository ?</div>

Basically, you can follow the instructions in [this repository](https://github.com/hysts/pytorch_image_classification) 
to configure the environment, 
but I added some other dependencies to requirements.txt. 
So in order to run these codes better, follow the steps below to install them.

<details open>
<summary>Installation</summary>

**Step 1: Conda environment create**

```bash
conda create -n img-class python=3.8
```

**Step 2: Pytorch install**

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3
```

**Step 3: Requirements.txt install**

```bash
pip install -r requirements.txt
```

</details>

# TODO: continuous update...