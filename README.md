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
> Kaggle 实验建议直接在竞赛 Notebook 环境中运行；MNIST 实验可在 LMCC、本地 Jupyter 或 VS Code 中运行。原始数据不纳入版本控制，获取入口见下表。

## 实验一览

| 实验 | 平台 | 任务 | 主要方法 | 数据  |
| :--- | :---: | :--- | :--- | :--- |
| [Lab 1-1 · PM2.5 预测](lab1-1/lab1-Kaggle-PM2.5.ipynb) | Kaggle | 根据连续 9 小时空气质量指标预测下一小时 PM2.5 | 特征标准化、线性回归、Adagrad | [2026zjutest1 课程竞赛](https://www.kaggle.com/competitions/2026zjutest1) |
| [Lab 1-2 · MNIST 手写数字识别](lab1-2/lab1-LMCC-手写数字识别.ipynb) | LMCC / Jupyter | 将手写数字图像分类为 0–9 | 全连接神经网络、交叉熵、Adam | [TorchVision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) |
| [Lab 2-1 · Adult Income](lab2-1/lab2-1-Kaggle-Adult-Income.ipynb) | Kaggle | 判断个人年收入是否超过 50K | 缺失值处理、类别编码、分层验证、梯度提升 | [UCI Adult 原始数据](https://archive.ics.uci.edu/dataset/2/adult) |


## 快速开始

Kaggle 实验可从表格中的竞赛入口加入数据，再导入对应 Notebook 并执行 **Run All**。平台输入位于 `/kaggle/input`，生成的提交文件位于 `/kaggle/working`。

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


