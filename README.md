<div align="center">

<h1>DeepLearning-ZJU</h1>

<p><strong>浙江大学《深度学习基础与实践》课程实验</strong></p>

<p>用可复现的 Notebook 记录建模、验证与提交过程。</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white">
  <img alt="Typst" src="https://img.shields.io/badge/Typst-239DAD?style=flat-square&logo=typst&logoColor=white">
</p>

</div>

## 项目简介

本仓库用于整理课程实验中的 Notebook、关键运行结果与竞赛提交文件。

> [!NOTE]
> Kaggle 实验建议直接在竞赛 Notebook 环境中运行；所有实验可在 [LMCC](http://122.226.162.86:8787/courses/57deb3d9-fd3c-4864-ac1c-daacb378defa)、本地 Jupyter 或 VS Code 中运行。原始数据不纳入版本控制，获取入口见下表。

- `LMCC`为浙大平台，若无权访问课程页可以使用：1.本地或其他在线Jupyter环境 2.在`实验一览`中找到数据的下载源，或者在notebook中找到下载链接

## 实验一览

> [!TIP]
> labx-1为kaggle平台，labx-2为LMCC平台

1. **[Lab 1-1 · PM2.5 预测](lab1-1/lab1-Kaggle-PM2.5.ipynb)**
   - 任务：根据连续 9 小时空气质量指标预测下一小时 PM2.5
   - 主要方法：特征标准化、线性回归、Adagrad
   - 数据：[2026zjutest1 课程竞赛](https://www.kaggle.com/competitions/2026zjutest1)

2. **[Lab 1-2 · MNIST 手写数字识别](lab1-2/lab1-LMCC-手写数字识别.ipynb)**
   - 任务：将手写数字图像分类为 0–9
   - 主要方法：全连接神经网络、交叉熵、Adam
   - 数据：[TorchVision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

3. **[Lab 2-1 · Adult Income](lab2-1/lab2-1-Kaggle-Adult-Income.ipynb)**
   - 任务：判断个人年收入是否超过 50K
   - 主要方法：缺失值处理、类别编码、分层验证、梯度提升
   - 数据：[UCI Adult 原始数据](https://archive.ics.uci.edu/dataset/2/adult)

4. **[Lab 2-2 · ASL 手语字母分类](lab2-2/02_asl.ipynb)**
   - 任务：识别 28×28 灰度手语图像对应的 24 个静态字母类别
   - 主要方法：自定义 Dataset、全连接神经网络、ReLU、交叉熵、Adam
   - 数据：[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

5. **[Lab 3-1 · Kaggle 表情图片分类](lab3-1/lab3-1-Kaggle-表情分类-CNN.ipynb)**
   - 任务：将 48×48 灰度人脸图像分为 7 类表情
   - 主要方法：CNN、BatchNorm、Dropout、类别加权交叉熵、AdamW、测试时增强
   - 数据：[2024-zjutest-3 课程竞赛](https://www.kaggle.com/competitions/2024-zjutest-3)

6. **[Lab 3-2 · ASL 卷积神经网络](lab3-2/03_asl_cnn.ipynb)**
   - 任务：使用 CNN 提高手语字母分类的验证集表现与泛化能力
   - 主要方法：Conv2d、BatchNorm、MaxPool2d、Dropout、全连接分类器
   - 数据：[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

7. **[Lab 4-1 · Kaggle 表情分类数据增强](lab4-1/lab4-Kaggle-表情分类-数据增强.ipynb)**
   - 任务：通过在线数据增强训练 CNN，并生成 7 类表情预测结果
   - 主要方法：随机翻转与仿射变换、亮度与对比度扰动、Random Erasing、AdamW、余弦退火、早停
   - 数据：[2024-zjutest4 课程竞赛](https://www.kaggle.com/competitions/2024-zjutest4)

8. **Lab 4-2 · ASL 数据增强与部署：[模型训练](lab4-2/04a_asl_augmentation.ipynb) / [推理预测](lab4-2/04b_asl_predictions.ipynb)**
   - 任务：增强 ASL 训练数据、保存模型，并对新的手语图片进行推理
   - 主要方法：自定义 CNN 模块、RandomResizedCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、模型保存与加载
   - 数据：[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)


## 快速开始

Kaggle 实验可从上述实验条目中的竞赛入口加入数据，再导入对应 Notebook 并执行 **Run All**。平台输入位于 `/kaggle/input`，生成的提交文件位于 `/kaggle/working`。

在 LMCC 或本地运行 Jupyter Notebook 时，可先创建独立环境并安装主要依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib torch torchvision scikit-learn
jupyter lab
```

MNIST Notebook 设置了 `download=True`，首次运行会自动下载数据。Kaggle Notebook 使用平台路径，本地复现时需要相应调整输入与输出目录。

## 复现约定

- 按照 Notebook 从上到下运行，避免依赖旧内核中残留的变量。
- 保留能够说明结果的训练日志、验证指标和提交文件，不提交可重复下载的原始数据。
- 调整模型或超参数时，同时记录修改内容与结果变化，避免只保留最好的一次。
- 提交前检查 CSV 的行数、列名、缺失值，以及 Notebook 中展示的结果是否一致。

