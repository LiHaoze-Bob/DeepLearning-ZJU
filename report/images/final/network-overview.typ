// 根据 CIFAR10TransferNet 和 TorchVision ConvNeXt-Tiny 实现绘制。
// 使用 Typst 原生矢量元素，文字、尺寸和连接关系均可直接编辑。
#let node(title, detail, tint: rgb("f3f6fa"), height: 5.5em) = block(
  width: 100%, height: height, inset: 5pt,
  fill: tint, stroke: 0.6pt + rgb("a7b3c2"), radius: 3pt,
  align(center + horizon)[
    #text(weight: "bold")[#title]
    #v(3pt)
    #detail
  ],
)

#let network-overview() = block(width: 100%, breakable: false)[
  #set text(size: 8.5pt, font: ("PingFang SC", "Helvetica Neue"))
  #set par(justify: false, first-line-indent: 0pt, leading: 0.35em, spacing: 0pt)
  #set align(center)
  #let arrow = text(size: 13pt, fill: rgb("586b7b"))[→]
  #let down = align(center, text(size: 13pt, fill: rgb("586b7b"))[↓])

  #grid(
    columns: (1.05fr, auto, 1.65fr, auto, 1.5fr),
    column-gutter: 4pt, align: horizon,
    node([CIFAR-10 输入], [RGB 图像\ N × 3 × 32 × 32]), arrow,
    node([训练增强（仅训练）], [裁剪、翻转、RandAugment\ CutMix 按概率启用]), arrow,
    node([net 内部预处理], [双线性缩放至 224 × 224\ ImageNet 均值 / 标准差]),
  )
  #v(3pt)
  #down
  #v(3pt)

  #block(width: 100%, inset: 7pt, stroke: 0.7pt + rgb("99adc4"), radius: 3pt)[
    #text(weight: "bold", size: 9pt)[ConvNeXt-Tiny 骨干 · ImageNet-1K 预训练初始化]
    #v(5pt)
    #grid(
      columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
      column-gutter: 3pt, align: horizon,
      node([Stem · s4], [4×4 卷积\ 96×56×56], height: 5.4em), arrow,
      node([Stage 1], [3 个残差块\ 96×56×56], height: 5.4em), arrow,
      node([Stage 2], [3 个残差块\ 192×28×28], height: 5.4em), arrow,
      node([Stage 3], [9 个残差块\ 384×14×14], height: 5.4em), arrow,
      node([Stage 4], [3 个残差块\ 768×7×7], height: 5.4em),
    )
    #v(5pt)
    #text(size: 8pt)[s4 表示步长 4；阶段间用 LayerNorm + 2×2、步长 2 下采样；尺寸为 C×H×W。]
  ]
  #v(3pt)
  #down
  #v(3pt)

  #grid(
    columns: (1fr, auto, 1.45fr, auto, 1fr),
    column-gutter: 4pt, align: horizon,
    node([全局平均池化], [空间均值：7×7 → 1×1\ 输出 768 维特征], tint: rgb("f1f7f3"), height: 4.9em), arrow,
    node([十分类头], [LayerNorm + Flatten\ Linear：768 → 10], tint: rgb("f1f7f3"), height: 4.9em), arrow,
    node([分类输出], [N × 10 logits\ 覆盖全部十类], tint: rgb("f1f7f3"), height: 4.9em),
  )
  #v(7pt)
  #block(width: 100%, inset: 6pt, fill: rgb("f7f7f7"), radius: 3pt)[
    #text(weight: "bold")[训练：先冻结骨干训练分类头 2 轮，再解冻全网络微调 9 轮。]
  ]
  #v(5pt)
  #block(width: 100%, inset: 6pt, fill: rgb("f1f7f3"), radius: 3pt)[
    #text(weight: "bold")[验证 / 推理：使用同一份 EMA 权重，关闭训练增强。]
    #v(4pt)
    原图 / 水平翻转图 → 同一骨干与分类头分别前向 → 两组 logits 平均 → 预测类别
  ]
]
