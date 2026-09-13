#import "template.typ": project, note, warning, analysis

#show: project.with(
  theme: "lab",
  course: "课程综合实践I-\n深度学习基础——理论与实践",
  title: "实验六实验报告",
  name: "实验六：循环神经网络RNN",
  author: "李昊泽",
  school_id: "3250102780",
  college: "计算机科学与技术学院",
  major: "计算机科学与技术",
  teacher: "王总辉",
  place: "紫金港海洋大楼11机房",
  date: "2026年9月7日",
  font_serif: ("Songti SC", "New Computer Modern"),
  font_sans_serif: ("PingFang SC", "Helvetica Neue"),
  font_mono: ("Monaco",),
)

// 补图方法：将对应 screenshot(...) 整段替换为
// #figure(image("images/lab6-01.png", width: 100%), caption: [对应图题])
// height 可按截图比例调整。本报告不依赖尚未提供的截图文件。
#let screenshot(title, suggestion, height: 5.5cm) = figure(
  block(
    width: 100%,
    height: height,
    fill: white,
    stroke: (paint: luma(165), thickness: 0.8pt, dash: "dashed"),
    radius: 4pt,
    inset: 14pt,
    align(center + horizon)[
      #text(size: 12pt, weight: "bold", fill: luma(85))[截图留空区]
      #v(0.8em)
      #text(size: 10pt, fill: luma(115))[#suggestion]
    ],
  ),
  kind: image,
  caption: title,
)

#set table(inset: 6pt)

= 摘要

Kaggle 部分使用两层 LSTMCell 学习随机相位正弦波的下一时刻数值，并自回归预测未来 1000 个时间步；完整训练后，训练 MSE 为 $4.1538 times 10^(-6)$，已知测试区间 MSE 为 $4.5628 times 10^(-6)$。
\
\
LMCC 部分从零实现字符级 RNN，在《The Time Machine》语料上比较顺序分区与随机采样。训练 500 轮后，两种方法的困惑度分别为 1.0245 和 1.5185。实验说明 RNN 能利用历史隐藏状态建模序列依赖，LSTM 能改善长序列中的信息保留。





= 实验原理

== 循环神经网络

普通前馈网络把每个输入独立处理，而 RNN 在时间步之间传递隐藏状态。对第 $t$ 个时间步，字符级 RNN 的计算为：

$ H_t = tanh(X_t W_(x h) + H_(t-1) W_(h h) + b_h) $

$ Y_t = H_t W_(h q) + b_q $

其中 $X_t$ 是当前输入，$H_(t-1)$ 保存此前序列的信息，$Y_t$ 是当前时刻的类别 logits。实验二先把字符索引转换为 one-hot 向量，再逐时间步更新隐藏状态。

== LSTM 门控机制

普通 RNN 在长序列反向传播时容易出现梯度消失或梯度爆炸。LSTM 在隐藏状态之外增加记忆单元 $C_t$：

$ f_t = sigma(W_f [h_(t-1), x_t] + b_f) $

$ i_t = sigma(W_i [h_(t-1), x_t] + b_i) $

$ o_t = sigma(W_o [h_(t-1), x_t] + b_o) $

$ tilde(C)_t = tanh(W_c [h_(t-1), x_t] + b_c) $

$ C_t = f_t dot C_(t-1) + i_t dot tilde(C)_t $

$ h_t = o_t dot tanh(C_t) $

遗忘门决定保留多少旧信息，输入门决定写入多少新信息，输出门决定当前记忆有多少参与隐藏状态。实验一使用两个串联的 LSTMCell，使第二层在每个时间步接收第一层提取的时序表示。

== 损失函数与评价指标

正弦波预测是回归任务，采用均方误差：

$ L_"MSE" = 1/N sum_(i=1)^N (hat(y)_i - y_i)^2 $

字符预测是多分类任务，使用交叉熵。若平均交叉熵为 $L_"CE"$，困惑度定义为：

$ "PPL" = exp(L_"CE") $

困惑度越低，表示模型对下一个字符越确定。当 PPL 接近 1 时，模型几乎能在当前数据上确定下一个字符，但仍需独立验证集才能判断泛化性能。

== 梯度裁剪与采样方法

长序列可能导致梯度范数快速增大。当全局梯度范数超过阈值 $theta$ 时，按统一比例缩放：

$ g <- theta / norm(g) dot g $

这样保留梯度方向，同时限制更新幅度。本实验阈值为 1.0。

顺序分区让相邻批次在原序列上连续，因此隐藏状态可以跨批次传递，但需要在下一批次前执行 `detach`，避免反向传播图无限延长。随机采样会打乱子序列，相邻批次通常不连续，因此每个批次都重新初始化隐藏状态。

#pagebreak()

= 实验一：Kaggle LSTM 正弦波预测

== 数据生成

设置 $T=20$，随机生成 100 条长度为 1000 的序列。第 $i$ 条序列使用随机整数偏移 $s_i in [-4T, 4T)$：

$ y_t^((i)) = sin((t+s_i)/T) $

不同序列频率相同，但起始相位不同。前 3 条序列作为测试数据，后 97 条作为训练数据。对每条序列，将 $[x_0, dots, x_998]$ 作为输入，将向后移动一位的 $[x_1, dots, x_999]$ 作为监督标签



== 网络结构

模型由两个 LSTMCell 和一个线性输出层构成：

#table(
  columns: (1.3fr, 1.5fr, 1.8fr),
  [*阶段*], [*输入与输出*], [*作用*],
  [LSTMCell 1], [`1 → 50`], [把当前标量与第一层历史状态编码为 50 维表示],
  [LSTMCell 2], [`50 → 50`], [进一步提取时序特征并传递第二层记忆],
  [Linear], [`50 → 1`], [预测下一时刻的信号值],
)

模型共有 31,051 个可训练参数。已知区间采用真实 $x_t$ 作为下一步输入；进入未来区间后，将上一时刻预测值重新送入第一层 LSTM，连续生成 1000 个未来点。这种方式属于自回归预测，误差可能随预测步数逐渐累积。

#figure(
  image(
    "../assets/27.png",
  ),
  caption: [训练曲线对比图]
)

#block(breakable: false)[
```python
for x_t in x.split(1, dim=1):
    h_t, c_t = self.layer1(x_t, (h_t, c_t))
    h_t2, c_t2 = self.layer2(h_t, (h_t2, c_t2))
    output = self.linear(h_t2)

for _ in range(future):
    h_t, c_t = self.layer1(output, (h_t, c_t))
    h_t2, c_t2 = self.layer2(h_t, (h_t2, c_t2))
    output = self.linear(h_t2)
```
]



== 训练设置

使用 MSELoss 和 LBFGS 优化器，学习率为 0.8，共训练 15 轮。LBFGS 的一次 `step` 会多次调用闭包，因此闭包内部必须依次完成清空梯度、前向传播、计算损失和反向传播。

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  [*训练轮次*], [*训练 MSE*], [*训练轮次*], [*训练 MSE*],
  [1], [`1.43358e-3`], [5], [`4.845e-5`],
  [10], [`5.62e-6`], [15], [`4.15385e-6`],
)

#figure(
  image(
    "../assets/28.png",
  ),
  caption: [训练曲线]
)



== 结果分析



训练 MSE 与测试 MSE 数量级接近，说明模型对未参与训练的随机相位序列也能完成准确的一步预测。预测图中，前 999 个点以实线表示，未来 1000 个点以点线表示；点线仍保持接近原信号的振幅、周期和相位连续性。
#pagebreak()

= 实验二：LMCC 字符级 RNN 语言模型

== 数据预处理

实验使用 《The Time Machine》文本。读取 3221 行原始文本，将非字母字符替换为空格并转为小写，再按字符切分。词表包含 28 个符号；正式实验使用语料的前 10,000 个字符。

设置 `batch_size=32`、`num_steps=35`。输入 $X$ 与标签 $Y$ 均为 `32×35`，且 $Y$ 相对 $X$ 向后移动一个字符。例如：





== 从零构建 RNN

模型不调用 `nn.RNN`，而是手动创建 $W_(x h)$、$W_(h h)$、$b_h$、$W_(h q)$ 和 $b_q$。输入与输出类别数均为 28，隐藏单元数为 512，总参数量为：

$ 28 times 512 + 512 times 512 + 512 + 512 times 28 + 28 = 291356 $

每个批次的 35 个时间步先转置并进行 one-hot 编码。模型把所有时间步和样本的输出拼接，因此输出形状为 `(35×32, 28)=(1120, 28)`；最终隐藏状态形状为 `(32, 512)`。



== 训练与文本生成

两个模型均采用交叉熵损失、手动 SGD、学习率 1.0、梯度裁剪阈值 1.0，并训练 500 轮。生成文本前先用给定前缀进行预热，之后每一步选择 logits 最大的字符，并将其作为下一时刻输入。

#block(breakable: false)[
```python
for character in prefix[1:]:
    _, state = net(get_input(), state)
    outputs.append(vocab[character])

for _ in range(num_preds):
    y, state = net(get_input(), state)
    outputs.append(int(y.argmax(dim=1).item()))
```
]

== 顺序采样

500 轮后
\

最终困惑度：`1.0244508`
\

处理速度：`43422.0 tokens/s`

最终生成：

#block(breakable: false)[
```text
time traveller for so it will be convenient to speak of himwas e
travelleryou can show black is white by argument said filby
```
]

困惑度接近 1，且生成内容形成较完整的英文短语。

#figure(
  image(
    "../assets/29.png",
  ),
  caption: [训练曲线]
)


== 随机采样

随机采样打乱固定长度子序列，并为每个批次重新初始化隐藏状态。500 轮后的正式结果为：

最终困惑度:`1.5184849`
\
处理速度:`54251.9 tokens/s`


最终生成：

#block(breakable: false)[
```text
time traveller but now you begin to seethe objecr of him frane i
traveller but now you begin to seethe objecr of him frane i
```
]

#figure(
  image(
    "../assets/30.png",
  ),
  caption: [训练曲线]
)

随机采样的处理速度比顺序分区高约 24.9%，但困惑度更高，生成文本中也容易有拼写错误。这是因为批次之间缺少连续隐藏状态，模型更难利用跨子序列上下文。




== 温度与 Top-k 

进一步实现了带温度和 Top-k 的概率采样。温度越低，分布越尖锐，输出更保守；温度越高，生成结果更多样，但拼写和语法错误也可能增加。Top-K 的意思是：从模型给出的所有候选结果中，选出得分最高的前 K 个。
\
\
`temperature=0.1`、`top_k=8` 
```text
the time traveller for so it will be convenient to speak of himwas expounding a recondite matter to us his grey eyes shone andtwinkled and his usually pale face was flushed and animated thefire....
```

`temperature=1.5`、`top_k=8` 
```text
the time traveller fareer alndth are der nstrantyeficc uf thetrldoking wave been at work upon thisgeometry of four dimensions for some time some of my resultsare curious for instance here is a portrait of a man at eightarant con th a dean...
```

`temperature=0.1`、`top_k=3` 
```text
the time traveller for so it will be convenient to speak of himwas expounding a recondite matter to us his grey eyes shone andtwinkled and his usually pale face was flushed and animated thefire burned brightly and the soft radiance of ...
```


= 讨论与改进

== 为什么多步预测会累积误差

一旦某一步产生误差，该误差会成为下一步输入，并通过隐藏状态继续传播。正弦波规律简单且无噪声，因此本次外推仍然稳定；复杂时间序列可采用滚动验证、噪声扰动训练或 scheduled sampling 缓解这一问题。

== 困惑度与生成质量的关系

困惑度衡量模型对真实下一个字符的平均不确定性，不直接衡量长文本是否语义连贯。贪心生成始终选择最高概率字符，可能出现重复；概率采样能够增加多样性，也可能引入错误。

== 实验局限
+ 正弦波数据由单一无噪声公式生成，可增加频率、振幅、趋势和噪声变化，检验模型鲁棒性。
+ 字符 RNN 只使用前 10,000 个字符且没有独立验证集，应划分训练集与验证集，保存验证困惑度最低的模型。
+ 当前文本预处理删除标点和换行，导致生成结果出现单词粘连；可保留更完整的字符表并显式建模空格、标点和换行。



= 参考资料

+ 课程资料：《RNN 实验》。
+ 本报告的部分文字组织、代码核对与排错分析由 OpenAI Codex 辅助完成。所有实验数据、运行输出和技术描述均由本人检查，最终报告由本人审阅并负责。
