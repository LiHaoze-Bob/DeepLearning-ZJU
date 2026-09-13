#import "template.typ": project

#show: project.with(
  theme: "lab",
  course: "课程综合实践I-\n深度学习基础——理论与实践",
  title: "实验八实验报告",
  name: "实验八：Transformer",
  author: "李昊泽",
  school_id: "3250102780",
  college: "计算机科学与技术学院",
  major: "计算机科学与技术",
  teacher: "王总辉",
  place: "紫金港海洋大楼11机房",
  date: "2026年9月9日",
  font_serif: ("Songti SC", "New Computer Modern"),
  font_sans_serif: ("PingFang SC", "Helvetica Neue"),
  font_mono: ("Monaco",),
)

// 补图方法：将对应 screenshot(...) 整段替换为
// #figure(image("images/lab8-01.png", width: 100%), caption: [对应图题])
// height 可按截图比例调整。本报告只保留截图区域，不依赖图片文件。
#let screenshot(title, suggestion, height: 5.3cm) = figure(
  block(
    width: 100%,
    height: height,
    fill: white,
    stroke: (paint: luma(165), thickness: 0.8pt, dash: "dashed"),
    radius: 4pt,
    inset: 14pt,
    breakable: false,
    align(center + horizon)[
      #text(size: 12pt, weight: "bold", fill: luma(85))[截图留空区]
      #v(0.8em)
      #text(size: 10pt, fill: luma(115))[#suggestion]
    ],
  ),
  kind: image,
  caption: title,
)

#let pending = text(fill: luma(100))[待修复并实际运行后填写]
#set table(inset: 6pt)

= 摘要

本实验包含 Transformer 正弦波预测和 BERT 自然语言处理两部分。


\
= 实验原理

== Transformer 与自注意力

Transformer 不依赖循环结构，而是通过注意力直接建模序列任意位置之间的关系。给定输入表示 $X$，先通过三个线性映射得到查询、键和值：

$ Q = X W_Q, quad K = X W_K, quad V = X W_V $

缩放点积注意力为：

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) V $

$Q K^T$ 衡量不同 token 之间的相关性，除以 $sqrt(d_k)$ 可避免维数较大时点积过大，Softmax 则把相关性转换为权重。多头注意力并行计算多组注意力，使不同子空间可以分别关注局部变化、长距离关系和周期模式：

$ "MultiHead"(Q,K,V) = "Concat"("head"_1, dots, "head"_h) W_O $

每个子层外部还包含残差连接和 Layer Normalization；前馈网络对每个位置的表示作相同的非线性变换。

== 位置编码

注意力本身不包含顺序信息，因此需要给每个位置加入位置编码。实验采用固定正余弦位置编码：

$ "PE"("pos", 2i) = sin("pos" / 10000^(2i/d)) $

$ "PE"("pos", 2i+1) = cos("pos" / 10000^(2i/d)) $

不同频率的正弦、余弦分量使模型能够区分位置，并表达相对距离。正弦波数值先经线性层由 1 维映射到 128 维，再与位置编码相加后送入 Transformer。

== Encoder-Decoder

Encoder 对已知的 `src` 序列执行自注意力，形成包含整体趋势、周期和相位信息的 memory。Decoder 先对已有的 `tgt` 前缀执行 masked self-attention，再通过 cross-attention 查询 Encoder memory，最后由线性层输出下一时刻的正弦值。

训练时使用真实历史值作为 Decoder 输入，这称为 teacher forcing。为防止模型在预测当前值时看到后面的真实值，需要在 Decoder 中使用上三角因果掩码：

$ M_(i,j) = cases(0 & j <= i, -infinity & j > i) $

推理时只给 Decoder 一个起始值，之后把每一步预测追加到输入中：

```text
真实 x79 → 预测 x80
预测 x80 → 预测 x81
预测 x81 → 预测 x82 → ...
```

训练使用均方误差：

$ L_"MSE" = 1/N sum_(i=1)^N (hat(y)_i - y_i)^2 $

teacher forcing 每一步都得到真实前值，自回归推理却会反复使用模型自身输出，因此后者更容易累积误差。

== BERT 与掩码语言模型

BERT 是仅使用 Transformer Encoder 的双向预训练模型。其输入表示由 token embedding、position embedding 和 token-type embedding 相加得到。与自回归 Decoder 只能查看左侧历史不同，BERT 的双向注意力可以同时利用掩码位置两侧的上下文。

\

掩码语言模型将某个 token 替换为 `[MASK]`。BERT 对该位置输出覆盖整个词表的 logits，再用 `argmax` 取得预测 token：

$ hat(t) = "argmax"_(v in V) z_v $

本实验模型词表大小为 28,996，因此长度为 24 的输入对应输出形状 $(1, 24, 28996)$。

== 抽取式问答

抽取式问答模型不直接生成新句子，而是为输入中的每个 token 分别计算“作为答案起点”和“作为答案终点”的分数。若输出为 $s_i$ 和 $e_i$，最简单的解码方式是：

$ i_"start" = "argmax"_i s_i, quad i_"end" = "argmax"_i e_i $

随后截取并解码区间 $[i_"start", i_"end"]$。实际系统还应约束终点不早于起点，并限制答案最大长度。


#pagebreak()
= 实验一：Kaggle Transformer 正弦波预测

== 数据生成

以 0.1 为间隔生成：

$ y_t = sin(0.1t), quad t = 0, 1, dots, 999999 $




== 网络结构

模型由输入映射、位置编码、Transformer 和输出层组成：

#table(
  columns: (1.5fr, 1.45fr, 2fr),
  [*模块*], [*尺寸或参数*], [*作用*],
  [Encoder 输入映射], [`Linear(1, 128)`], [把标量信号映射到高维特征],
  [Decoder 输入映射], [`Linear(1, 128)`], [编码已知或已预测的目标前缀],
  [位置编码], [`max_len=5000`], [给序列加入时间位置信息],
  [Transformer Encoder], [2 层、8 头], [提取已知序列的全局依赖],
  [Transformer Decoder], [2 层、8 头], [结合目标前缀和 Encoder memory],
  [前馈层], [$128 arrow 512 arrow 128$], [对每个位置作非线性特征变换],
  [输出层], [`Linear(128, 1)`], [还原为正弦信号预测值],
)

模型共有 926,849 个可训练参数，Dropout 为 0.1。

== 训练设置与结果

#table(
  columns: (1.1fr, 1fr, 1.1fr, 1fr),
  [*参数*], [*取值*], [*参数*], [*取值*],
  [随机种子], [42], [计算设备], [CPU],
  [batch size], [512], [训练轮数], [10],
  [隐藏维度], [128], [注意力头数], [8],
  [Encoder/Decoder 层数], [2/2], [学习率], [$10^(-3)$],
  [损失函数], [MSELoss], [优化器], [Adam],
  [训练/测试样本数], [8000/2000], [梯度裁剪阈值], [1.0],
)

各轮实际训练 MSE 为：

#table(
  columns: (0.6fr, 1fr, 0.6fr, 1fr, 0.6fr, 1fr),
  [*轮次*], [*训练 MSE*], [*轮次*], [*训练 MSE*], [*轮次*], [*训练 MSE*],
  [1], [1.545887], [2], [0.131543], [3], [0.027146],
  [4], [0.013253], [5], [0.009052], [6], [0.007009],
  [7], [0.005745], [8], [0.005701], [9], [0.016929],
  [10], [0.015096], [], [], [], [],
)




#figure(
  image(
    "../assets/34.png",
  ),
  caption: [预测结果],
)



teacher-forcing 预测能够较好贴近真实目标，而自回归误差约为前者的 20.7 倍。这说明模型具备较好的单步拟合能力，但连续使用自身预测后误差会传播。


#pagebreak()
= 实验二：LMCC BERT 自然语言处理

== 模型加载



```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
masked_lm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
masked_lm_model.eval()
```



== WordPiece 分词

输入文本为：

```text
I understand equations, both the simple and quadratical.
What kind of equations do I understand?
```

编码结果共 24 个 token。关键分词如下：

```text
[CLS] I understand equations , both the simple and
q ##uad ##ratic ##al . [SEP]
What kind of equations do I understand ? [SEP]
```

与按空格切分相比，token 数更多，是因为模型加入了 `[CLS]`、`[SEP]` 等特殊标记，并把低频词拆成多个 WordPiece 子词。



== 文本分段

第一段正文使用 0，第二段正文使用 1；两个 `[SEP]` 作为边界。当前实际输出为一个形状为 `1×24` 的张量：

```text
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```



== 掩码语言模型

实验把位置 5 的 `both` 替换为 `[MASK]`：

```text
[CLS] I understand equations, [MASK] the simple and quadratical.
[SEP] What kind of equations do I understand? [SEP]
```

模型利用左右两侧上下文成功恢复了缺失词。

#screenshot(
  [BERT 掩码词预测结果],
  [截取 embedding_table.shape、predictions logits 的 shape、predicted_index=1241 和 predicted_token='both'。],
  height: 5.5cm,
)

== 问题回答

目标是把原陈述和问题共同输入 SQuAD 微调后的 BERT，分别获得每个 token 的 `start_logits` 与 `end_logits`，再截取最高分的答案区间。






= 讨论与改进

== 为什么两种正弦波测试误差差异较大

teacher forcing 使用真实的 $x_t$ 预测 $x_(t+1)$，每一步输入都处在训练分布内；自回归推理则使用 $hat(x)_t$ 预测 $hat(x)_(t+1)$。若第一个预测产生误差 $epsilon_1$，后续输入分布就开始偏离真实数据，误差会继续影响注意力和解码输出。可尝试降低学习率、保存验证误差最低的模型、向 Decoder 输入加入噪声，或使用 scheduled sampling 缓解训练与推理之间的分布差异。

== Transformer 与 BERT 的注意力方向

正弦波实验中的 Encoder 可以双向观察整个 `src`；Decoder 必须使用 causal mask，只允许观察已生成的目标前缀。BERT 的 masked-LM 则需要结合 `[MASK]` 左右两侧信息，因此采用双向 Encoder 注意力。两者都使用注意力机制，但可见信息范围由任务目标决定。




= 实验结论

本实验验证了 Transformer 对序列依赖的建模能力。Encoder-Decoder 模型能够从正弦波前缀预测后续信号，但 teacher forcing 与自回归 MSE 的明显差距说明多步生成存在误差累积。BERT 实验展示了 WordPiece、特殊 token、segment ids、双向上下文和 masked-LM 的协同作用，模型成功恢复被遮盖的 `both`。


= 参考资料

+ 课程资料
+ 本报告的部分文字组织、代码核对与排错分析由 OpenAI Codex 辅助完成。所有实验数据、运行输出和技术描述均由本人检查，最终报告由本人审阅并负责。
