<div align="center">

<h1>DeepLearning-ZJU</h1>

<p><strong>浙江大学《深度学习基础与实践》课程实验</strong></p>

<p>用可复现的 Notebook 记录从基础网络到生成模型、Transformer 与课程设计的完整实践。</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white">
  <img alt="Typst" src="https://img.shields.io/badge/Typst-239DAD?style=flat-square&logo=typst&logoColor=white">
</p>

</div>

## 项目简介

本仓库整理了课程 Lab 1–8、课程设计、实验报告与必要的结果图。每个实验以 Notebook 为主要入口，报告同时保留可编辑的 Typst 源文件与导出的 PDF。

实验通常分为两类：`labx-1` 对应 Kaggle 或课程任务，`labx-2` 对应 LMCC 教学实验。Lab 8-1 虽保留 Kaggle 命名，但按课件要求只完成正弦波预测，不生成竞赛提交文件。

> [!NOTE]
> 文件存在只表示实验材料已经整理进项目，不等同于当前环境下完成了一次正式全量训练。部分耗时 Notebook 提供快速检查模式；快速运行只能验证代码链路，不能作为正式实验结果。

## 目录结构

```text
.
├── lab1-1 ... lab8-2/   # 课程实验 Notebook、少量代码与结果样例
├── final/               # CIFAR-10 课程设计与历史方案
├── report/              # Lab 1–8、课程设计的 Typst 源文件与 PDF
├── assets/              # 报告与 README 使用的公共图片
├── .gitignore           # 数据、缓存、模型权重与中间输出规则
└── README.md
```

原始数据、模型权重、运行缓存和大批量生成结果保留在本地，不纳入版本控制。`final/archive/` 保存课程设计的历史方案，主入口始终是 `final/3250102780_李昊泽.ipynb`。

## 实验一览

### Lab 1 · 线性回归与手写数字识别

1. **[Lab 1-1 · PM2.5 预测](lab1-1/lab1-Kaggle-PM2.5.ipynb)**
   - 根据连续 9 小时空气质量指标预测下一小时 PM2.5。
   - 主要方法：滑动窗口、特征标准化、线性回归、Adagrad。
   - 数据：[2026zjutest1 课程竞赛](https://www.kaggle.com/competitions/2026zjutest1)。

2. **[Lab 1-2 · MNIST 手写数字识别](lab1-2/lab1-LMCC-手写数字识别.ipynb)**
   - 将 28×28 手写数字图像分类为 0–9。
   - 主要方法：全连接神经网络、交叉熵、Adam。
   - 数据：[TorchVision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。

报告：[Typst 源文件](report/lab1.typ) · [PDF](report/lab1-report.pdf)

### Lab 2 · 全连接神经网络

1. **[Lab 2-1 · Adult Income](lab2-1/lab2-1-Kaggle-Adult-Income.ipynb)**
   - 根据人口统计特征判断个人年收入是否超过 50K。
   - 主要方法：分层划分、特征标准化、`106 → 64 → 64 → 2` 全连接网络、最佳权重保存。
   - 数据：[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)。

2. **[Lab 2-2 · ASL 手语字母分类](lab2-2/02_asl.ipynb)**
   - 识别 28×28 灰度手语图像对应的 24 个静态字母类别。
   - 主要方法：自定义 Dataset、全连接网络、ReLU、交叉熵、Adam。
   - 数据：[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)。

报告：[Typst 源文件](report/lab2.typ) · [PDF](report/lab2-report.pdf)

### Lab 3 · 卷积神经网络

1. **[Lab 3-1 · Kaggle 表情图片分类](lab3-1/lab3-1-Kaggle-表情分类-CNN.ipynb)**
   - 将 48×48 灰度人脸图像分为 7 类表情。
   - 主要方法：CNN、BatchNorm、Dropout、类别加权交叉熵、AdamW、测试时增强。
   - 数据：[2024-zjutest-3 课程竞赛](https://www.kaggle.com/competitions/2024-zjutest-3)。

2. **[Lab 3-2 · ASL 卷积神经网络](lab3-2/03_asl_cnn.ipynb)**
   - 使用 CNN 提高手语字母分类的验证表现与泛化能力。
   - 主要方法：Conv2d、BatchNorm、MaxPool2d、Dropout、全连接分类器。

报告：[Typst 源文件](report/lab3.typ) · [PDF](report/lab3-report.pdf)

### Lab 4 · 数据增强与模型部署

1. **[Lab 4-1 · Kaggle 表情分类数据增强](lab4-1/lab4-Kaggle-表情分类-数据增强.ipynb)**
   - 通过在线数据增强训练 CNN，并生成 7 类表情预测结果。
   - 主要方法：随机翻转与仿射变换、亮度与对比度扰动、Random Erasing、AdamW、余弦退火、早停。
   - 数据：[2024-zjutest4 课程竞赛](https://www.kaggle.com/competitions/2024-zjutest4)。

2. **Lab 4-2 · ASL 数据增强与部署**
   - Notebook：[模型训练](lab4-2/04a_asl_augmentation.ipynb) · [推理预测](lab4-2/04b_asl_predictions.ipynb)。
   - 主要方法：RandomResizedCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、模型保存与加载。

报告：[Typst 源文件](report/lab4.typ) · [PDF](report/lab4-report.pdf)

### Lab 5 · 迁移学习

1. **[Lab 5-1 · Kaggle 车辆图片分类](lab5-1/lab5-Kaggle-车辆分类-迁移学习.ipynb)**
   - 比较普通 CNN 与 ImageNet 预训练 VGG16-BN，并使用微调模型生成提交文件。
   - 主要方法：冻结分类头训练、解冻最后卷积块、小学习率微调、测试时翻转。

2. **Lab 5-2 · 预训练模型与个性化宠物门**
   - Notebook：[VGG16 直接推理](lab5-2/05a_doggy_door.ipynb) · [Penny 柯基迁移学习](lab5-2/05b_corgi_door.ipynb)。
   - 主要方法：ImageNet 预训练、输入预处理、冻结特征层、二分类迁移学习与微调。
   - 数据：[Penny the Corgi](https://www.kaggle.com/datasets/danielledetering/penny-the-corgi)。

报告：[Typst 源文件](report/lab5.typ) · [PDF](report/lab5-report.pdf)

### Lab 6 · 循环神经网络

1. **[Lab 6-1 · LSTM 正弦波预测](lab6-1/lab6-Kaggle-RNN-正弦波预测.ipynb)**
   - 使用两层 LSTMCell 学习下一时刻信号，并自回归外推未来序列。
   - 主要方法：合成正弦波、MSE、LBFGS、已知区间预测与多步外推。

2. **[Lab 6-2 · 字符级 RNN 语言模型](lab6-2/RNN_LMCC.ipynb)**
   - 从零实现 RNN，在 *The Time Machine* 语料上训练字符级语言模型。
   - 主要方法：one-hot、手写循环计算、梯度裁剪、顺序分区与随机采样、困惑度。

报告：[Typst 源文件](report/lab6.typ) · [PDF](report/lab6-report.pdf)

### Lab 7 · 生成对抗网络

1. **[Lab 7-1 · DCGAN 动漫人脸生成](lab7-1/lab7-Kaggle-GAN-动漫人脸生成.ipynb)**
   - 在动漫人脸数据上训练 DCGAN，生成 64×64 RGB 图像。
   - 主要方法：卷积判别器、转置卷积生成器、固定噪声观察、checkpoint 与批量导出。

2. **[Lab 7-2 · 条件 GAN 手写数字生成](lab7-2/GAN_LMCC.ipynb)**
   - 在 MNIST 上训练 Conditional GAN，并按给定标签生成指定数字。
   - 主要方法：标签嵌入、BCELoss、Adam、固定标签样例。

报告：[Typst 源文件](report/lab7.typ) · [PDF](report/lab7-report.pdf)

### Lab 8 · Transformer 与 BERT

1. **[Lab 8-1 · Transformer 正弦波预测](lab8-1/lab8-1-Kaggle-Transformer-正弦波预测.ipynb)**
   - 使用 Encoder-Decoder Transformer 根据序列前 80% 预测后 20%。
   - 主要方法：正余弦位置编码、多头注意力、teacher forcing、递归推理与 MSE 对比。

2. **[Lab 8-2 · BERT 自然语言处理](lab8-2/06_nlp.ipynb)**
   - 学习 WordPiece 分词、句子分段、掩码词预测与抽取式问答。
   - 首次运行需要联网下载预训练模型，或提前准备本地缓存。

报告：[Typst 源文件](report/lab8.typ) · [PDF](report/lab8-report.pdf)

### 课程设计 · CIFAR-10 图像分类

- 主 Notebook：[final.ipynb](final/final.ipynb)。
- 当前方案：TorchVision ConvNeXt-Tiny 迁移学习，两阶段微调，结合 RandAugment、CutMix、EMA 与测试时增强。
- 选择指标：飞机、猫、青蛙三类准确率的平均值，总体准确率作为平局判据。
- 报告：[Typst 源文件](report/final.typ) · [PDF](report/final-report.pdf)。

## 快速开始

### Kaggle

打开对应竞赛或课程 Notebook，添加数据后执行 **Run All**。Kaggle 输入通常位于 `/kaggle/input`，生成的提交文件和模型位于 `/kaggle/working`。运行前先阅读 Notebook 顶部的路径说明，不同竞赛的数据目录和 CSV 表头并不相同。

### 本地 Jupyter

建议使用独立环境。大多数实验需要以下依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas matplotlib pillow scikit-learn torch torchvision tqdm transformers
jupyter lab
```

不同 Notebook 的历史内核来自 Python 3.11 或 3.12；新环境不必严格复刻内核名称，但应确保 PyTorch、TorchVision 与 Transformers 彼此兼容。Lab 7 和课程设计训练量较大，本地 CPU 更适合做流程检查，正式训练建议使用 GPU。

### Typst 报告

安装 [Typst](https://typst.app/) 后，可从仓库根目录编译任一报告：

```bash
typst compile report/lab8.typ report/lab8-report.pdf
```

报告中的截图与指标应以对应 Notebook 的正式运行结果为准，不要把快速检查结果写成正式结论。

## 数据与产物约定

- `data/`、`asl_data/`：本地数据集，不提交。
- `*.pth`、`*.pt`、`*.npy`：模型权重、断点或生成数据，不提交。
- `.cache/`、`.ipynb_checkpoints/`、`__pycache__/`：运行缓存，不提交。
- `outputs/`、`checkpoints/`：训练中间产物与批量生成结果，不提交。
- `submit.csv`、`submission.csv`、`sample.csv`：体积较小的提交结果或格式样例，可以保留。
- `report/*.typ` 与 `report/*.pdf`：报告源文件与最终可读版本，可以保留。

## 复现检查

- 从空内核按顺序执行 Notebook，避免依赖旧会话中的变量。
- 正式运行前确认数据路径、类别映射、训练轮数和快速模式开关。
- 保存模型后重新创建网络、加载 `state_dict` 并切换到 `eval()`，确认推理接口可恢复。
- 提交前检查 CSV 的行数、列名、缺失值与样例格式。
- 区分流程检查、正式训练、课程验证集指标与 Kaggle 成绩；它们不是同一种证据。
