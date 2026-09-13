# 课程设计报告图片来源

这些图片供 `report/final.typ` 引用，不是新一轮训练结果。所有正式结果来自 `final/3250102780_李昊泽.ipynb` 中 2026-09-12 的 Kaggle 已保存输出。

- `network-overview.typ`：根据 Notebook 中 `CIFAR10TransferNet` 及 TorchVision 的 ConvNeXt-Tiny 结构绘制的可编辑矢量示意图，不是实验截图。包含单次前向、两阶段训练说明和同一 EMA 模型的双视图 TTA 路径。

- `learning-curves.png`：Notebook 第 22 节（按 JSON 从 0 开始计数的 cell 46）的第一张内嵌 PNG，直接 Base64 解码，未重新绘图或修改。
- `confusion-matrix.png`：同一单元格的第二张内嵌 PNG，直接解码，未重新绘图或修改。
- `verification-output.png`：将第 21 节的完整文本输出，以及第 24 节从重建 `net` 到断言检查的连续代码片段、完整文本输出排为 HTML 后截图。文字来自原 Notebook；不是 Kaggle 网站界面截图，也不表示新一次执行或成功提交。

原始图像 SHA-256：

- 学习曲线：`ace62e664801be4cb79e3b89295d1d278349f04188460e3c11a8ac3236f07045`
- 混淆矩阵：`494d8c1862a173359a1a5dc8451138415e13a715a288dd91b48fe4bdc8a0d127`

阅读时注意：最终公开验证集总体准确率为 98.67%，飞机、猫、青蛙平均为 98.63%。模型训练 2 轮分类头和 9 轮微调，30 轮仅是保留的学习率调度参考长度。重载一致性检查使用 2 张真实验证图像，不等于完整的新内核运行或隐藏测试。
