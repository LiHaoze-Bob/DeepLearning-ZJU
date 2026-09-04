#import "template.typ": project, figurex, note, tip, warning, important, analysis

#show: project.with(
  theme: "lab",
  course: "课程综合实践I-
深度学习基础——理论与实践",
  title: "实验三实验报告",
  name: "实验三：卷积神经网络",
  author: "李昊泽",
  school_id: "3250102780",
  college: "计算机科学与技术学院",
  major: "计算机科学与技术",
  teacher: "王总辉",
  place: "紫金港海洋大楼11机房",
  date: "2026年9月2日",
  font_serif: ("Songti SC", "New Computer Modern"),
  font_sans_serif: ("PingFang SC", "Helvetica Neue"),
  font_mono: ("Monaco",),
)

#let screenshot(title, suggestion, height: 5cm) = figure(
  block(
    width: 100%,
    height: height,
    fill: luma(246),
    stroke: 0.8pt + luma(155),
    radius: 4pt,
    inset: 14pt,
    align(center + horizon)[
      #text(size: 12pt, weight: "bold", fill: luma(65))[截图占位：#title]
      #v(0.8em)
      #text(size: 10pt, fill: luma(105))[建议插入：#suggestion]
    ],
  ),
  kind: image,
  caption: title,
)

= 摘要

本实验使用卷积神经网络完成两个图像分类任务。第一部分在 Kaggle 平台完成 48×48 灰度人脸表情七分类。模型由三组卷积、批归一化、RReLU 与最大池化构成，并使用分层验证、数据增强、类别加权交叉熵、AdamW、学习率调度和测试时增强。模型在 3300 个验证样本上的最佳准确率为 57.48%，在 Kaggle 测试集上的 Accuracy Score 为 0.58679。
\
\
第二部分完成美国手语图像 24 分类。模型将 28×28 灰度图像依次变换为 25、50、75 通道的特征图，再经全连接层输出类别 logits。20 轮训练中，训练准确率最高达到 100.00%，验证准确率最高达到 98.30%。


= 实验原理

== CNN 与全连接网络

卷积神经网络使用较小的卷积核在图像上滑动，同一卷积核在不同位置共享参数，能够学习边缘、纹理、局部轮廓及更高层语义特征。
\
\
对于二维输入特征图 $X$，第 $k$ 个卷积核在位置 $(i,j)$ 的输出可写为：

$ Y_(k,i,j) = b_k + sum_c sum_u sum_v W_(k,c,u,v) X_(c,i+u,j+v) $

卷积输出尺寸由输入尺寸 $H_("in")$、卷积核大小 $K$、步长 $S$ 和填充 $P$ 决定：

$ H_("out") = floor((H_("in") + 2P - K) / S) + 1 $

本实验使用 3×3 卷积核、步长 1 和填充 1，因此卷积前后的空间尺寸保持不变；随后由 2×2、步长 2 的最大池化将高和宽减半。

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  figure(
    image("../lab3-2/conv2d.png", width: 100%),
    caption: [卷积核在局部区域上提取特征],
  ),
  figure(
    image("../lab3-2/maxpool2d.png", width: 100%),
    caption: [2×2 最大池化示意],
  ),
)

== 激活、归一化



Kaggle 表情模型使用 RReLU。它在负半轴保留一个随机小斜率，避免负输入的梯度恒为 0。模型输出层不添加 Softmax，而是直接给出 logits，因为 `CrossEntropyLoss` 已在内部完成 LogSoftmax 与负对数似然计算。
\
\
BatchNorm 对一个 mini-batch 中同一通道的激活进行标准化，并学习缩放和平移参数，使不同层的数值尺度更稳定。Dropout 在训练阶段随机将一部分激活置零，迫使网络不依赖少数神经元；在 `model.eval()` 下会自动关闭。

#figure(
  image("../lab3-2/dropout.png", width: 78%),
  caption: [不同 Dropout 比例下的随机失活示意],
)

== 损失函数

设输出 logits 为 $z_0, z_1, ..., z_(C-1)$，Softmax 概率为：

$ p_k = exp(z_k) / sum_(j=0)^(C-1) exp(z_j) $

真实类别为 $y$ 时，单个样本的交叉熵损失为：

$ L = -log p_y $



= 实验一：Kaggle 表情图片分类

== 任务与数据

本任务将人脸表情划分为七类：0 生气、1 厌恶、2 恐惧、3 高兴、4 难过、5 惊讶、6 中立。每张图像以一个包含 2304 个灰度值的字符串存储在 `feature` 列中，对应 48×48 像素。



训练标签分布：

#table(
  columns: (0.8fr, 1fr, 0.8fr, 1fr, 0.8fr, 1fr, 0.8fr, 1fr),
  inset: 5pt,
  [*类别*], [*数量*], [*类别*], [*数量*], [*类别*], [*数量*], [*类别*], [*数量*],
  [0], [3062], [1], [355], [2], [3188], [3], [5529],
  [4], [3692], [5], [2375], [6], [3797], [], [],
)

#note[类别 1 只有 355 个样本，而类别 3 有 5529 个样本，类别分布差异明显。因此训练时采用类别加权交叉熵，避免模型只偏向样本较多的类别。]



== 数据预处理

使用空格分隔 `feature` 中的像素值，检查每行均含 2304 个数，再重塑为 `(N,1,48,48)` 的 PyTorch 张量。像素除以 255 后映射到 0～1，随后用均值 0.5、标准差 0.5 归一化到约 -1～1。
\
\
固定随机种子 42，并按标签分层抽取 15% 作为验证集，得到 18698 个训练样本、3300 个验证样本和 6711 个测试样本。

== 网络结构

模型使用三段 `Conv2d → BatchNorm2d → RReLU → MaxPool2d`。

#table(
  columns: (1.45fr, 1.9fr, 1.6fr),
  inset: 6pt,
  [*阶段*], [*输入与输出*], [*主要参数*], 
  [卷积块 1], [`1×48×48 → 64×24×24`], [`3×3, P=1`], 
  [卷积块 2], [`64×24×24 → 128×12×12`], [`3×3, P=1`], 
  [卷积块 3], [`128×12×12 → 256×6×6`], [`3×3, P=1`], 
  [Flatten], [`256×6×6 → 9216`], [-], 
  [全连接层], [`9216 → 512 → 128 → 7`], [Dropout 0.3/0.5], 
)

\
\
该模型共有 5,156,807 个可训练参数。课程资料中的参考结构使用更宽的 4096、1024 和 256 维全连接层；本实验缩小全连接层，以减少参数量和过拟合风险，同时保留三段卷积主干。

#figure(
  image(
    "../assets/13.png",
  ),
  caption: [模型结构],
)

== 训练策略

Kaggle 实验使用 Tesla T4 和 PyTorch 2.10.0+cu128，批大小为 256，共训练 30 个 epoch。损失函数为带类别权重和 0.05 标签平滑的交叉熵；优化器为 AdamW，初始学习率 0.001，权重衰减为 $10^(-4)$。当验证准确率连续两轮没有改善时，学习率减半；同时设置梯度裁剪、混合精度训练和早停，并按照最高验证准确率保存 `best_expression_cnn.pth`。



== 实验结果与分析

选取训练过程中的代表性轮次如下：

#table(
  columns: (0.7fr, 1.15fr, 1fr, 1.15fr, 1fr),
  inset: 6pt,
  [*Epoch*], [*训练 Loss*], [*训练 Acc*], [*验证 Loss*], [*验证 Acc*],
  [1], [2.1937], [19.94%], [1.8974], [35.48%],
  [5], [1.7367], [36.65%], [1.6534], [43.79%],
  [10], [1.6055], [44.37%], [1.5559], [49.15%],
  [17], [1.5037], [49.05%], [1.4850], [54.09%],
  [23], [1.4270], [52.06%], [1.4301], [56.12%],
  [27], [1.4010], [53.53%], [1.4139], [57.18%],
  [29], [1.3982], [53.59%], [1.4158], [57.48%],
  [30], [1.3910], [54.48%], [1.4036], [56.73%],
)

#figure(
  image(
    "../assets/14.png",
  ),
  caption: [训练结果],
)

#figure(
  image(
    "../assets/15.png",
  ),
  caption: [训练曲线],
)
\
\
\
最佳模型在验证集上的分类结果如下：

#table(
  columns: (1.1fr, 0.85fr, 0.85fr, 0.85fr, 0.8fr),
  inset: 5pt,
  [*类别*], [*Precision*], [*Recall*], [*F1*], [*Support*],
  [Angry], [0.5501], [0.4423], [0.4903], [459],
  [Disgust], [0.1780], [0.6415], [0.2787], [53],
  [Fear], [0.4681], [0.3536], [0.4029], [478],
  [Happy], [0.7966], [0.7976], [0.7971], [830],
  [Sad], [0.4874], [0.4206], [0.4516], [554],
  [Surprise], [0.6532], [0.7247], [0.6871], [356],
  [Neutral], [0.5007], [0.5930], [0.5430], [570],
)
\
\
#figure(
  image(
    "../assets/16.png",
  ),
  caption: [混淆矩阵],
)
Happy 的 F1 为 0.7971，是识别效果最好的类别；Surprise 次之。Disgust 只有 53 个验证样本，类别加权使其召回率达到 0.6415，但精确率仅为 0.1780，说明模型会把一部分其他表情误判为 Disgust。Fear 和 Sad 的 F1 也较低，反映出这些表情在低分辨率灰度图片中的区分难度较高。
\
\
\
\
最终Accuracy Score 为 *0.58679*，即约 58.68%。



#pagebreak()

= 实验二：美国手语图片识别（卷积version）

== 任务

本任务的数据处理与lab2基本相同：每张图像为 28×28 的单通道灰度图像，CSV 的第一列为 `label`，其余 784 列为像素。J 和 Z 需要连续运动才能表达，因此静态数据集中不包含这两个字母；本地数据已将其余类别连续映射为 0～23，共 24 类。




== 网络结构

ASL 模型包含三层卷积，每层使用 3×3 卷积、步长 1、填充 1，卷积后接 BatchNorm 和 ReLU。池化将图像尺寸从 28×28 依次减半到 14×14、7×7 和 3×3。第二个卷积块加入 0.2 Dropout，全连接层前加入 0.3 Dropout。

#figure(
  image("../lab3-2/cnn.png", height: 12cm),
  caption: [ASL CNN 整体结构],
)

#table(
  columns: (1fr, 1.8fr, 2.25fr, 1.75fr),
  inset: 4pt,
  text(size: 9pt)[*阶段*], text(size: 9pt)[*输入与输出*], text(size: 9pt)[*层配置*], text(size: 9pt)[*作用*],
  text(size: 9pt)[卷积块 1], text(size: 9pt)[1×28×28 → 25×14×14], text(size: 9pt)[Conv + BN + ReLU + Pool], text(size: 9pt)[提取基础边缘特征],
  text(size: 9pt)[卷积块 2], text(size: 9pt)[25×14×14 → 50×7×7], text(size: 9pt)[Conv + BN + ReLU + Dropout + Pool], text(size: 9pt)[组合局部手指轮廓],
  text(size: 9pt)[卷积块 3], text(size: 9pt)[50×7×7 → 75×3×3], text(size: 9pt)[Conv + BN + ReLU + Pool], text(size: 9pt)[形成更高层手势特征],
  text(size: 9pt)[Flatten], text(size: 9pt)[75×3×3 → 675], text(size: 9pt)[-], text(size: 9pt)[转为一维特征向量],
  text(size: 9pt)[全连接], text(size: 9pt)[675 → 512 → 24], text(size: 9pt)[Dropout 0.3], text(size: 9pt)[输出 24 类 logits],
)

该网络共有 404099 个可训练参数。与上一实验的全连接 ASL 模型相比，它在展平前保留二维空间关系，更适合识别手指边缘、局部轮廓和不同手势的相对位置。

#figure(
  image(
    "../assets/12.png"
  ),
  caption: [模型结构],
)

== 训练与验证

模型使用 `CrossEntropyLoss` 和 Adam 默认参数训练 20 个 epoch。训练函数在 `model.train()` 下依次执行前向传播、清空梯度、计算损失、反向传播和参数更新；验证函数在 `model.eval()` 与 `torch.no_grad()` 下只进行前向传播。

```python
def train():
    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

def validate():
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
```



== 实验结果与分析

选取代表性训练结果如下。Notebook 中 epoch 从 0 开始编号：

#table(
  columns: (0.75fr, 1.2fr, 1fr, 1.2fr, 1fr),
  inset: 6pt,
  [*Epoch*], [*训练 Loss*], [*训练 Acc*], [*验证 Loss*], [*验证 Acc*],
  [0], [268.6069], [90.74%], [25.1972], [95.71%],
  [1], [16.9970], [99.49%], [19.8946], [97.24%],
  [2], [12.3731], [99.65%], [11.4613], [97.91%],
  [7], [1.4564], [99.96%], [17.0607], [97.84%],
  [12], [9.1179], [99.72%], [10.9429], [98.09%],
  [15], [0.3725], [99.99%], [14.1010], [98.30%],
  [16], [0.0547], [100.00%], [12.5894], [98.28%],
  [17], [0.0452], [100.00%], [16.7879], [98.26%],
  [18], [0.1935], [99.99%], [400.2440], [73.70%],
  [19], [14.9442], [99.54%], [257.0182], [83.41%],
)

验证准确率在 Epoch 15 达到最高值 98.30%，Epoch 16 和 17 仍保持在 98.2% 左右；但 Epoch 18 突然降至 73.70%，最后一轮仅恢复到 83.41%。同时训练准确率几乎一直保持在 100%，说明模型已经充分拟合训练数据，而验证表现存在明显波动。
\
\
从现有证据可以确认：CNN 相比上一实验的全连接模型显著提高了 ASL 验证准确率，证明保留二维局部结构对手势识别有效。



= 参考资料

+ 课程资料
+ Kaggle Competition：2024ZJUTEST3，Deep Learning class Experiment 3。
+ NVIDIA Deep Learning Institute，`03_asl_cnn.ipynb` 美国手语 CNN 实验。
+ 本报告的部分文字组织、代码核对与排错分析由 OpenAI Codex 辅助完成。所有实验数据、运行输出和技术描述均由本人检查，最终报告由本人审阅并负责。
