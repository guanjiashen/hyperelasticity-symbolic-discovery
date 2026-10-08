# 数据来源

`experimental/` 保留模型发现工作目录中的名义应力与变形数值输入。
`stress_stretch.txt` 的列顺序为名义应力在前、拉伸比（或剪切量）在后。
使用前请确认各加载模式的约定。数据加载器可以插入无应力参考点，这些点不应计入实验测量点数。

- `Treloar_1944`：Treloar（1944）的硫化橡胶曲线数字化数据，单位 MPa。
- `Meunier_2008`：Meunier 等（2008）的硅橡胶曲线数字化数据，单位 MPa。
- `Yohsuke_2011`：Bitoh 等（2011）的聚合物凝胶曲线数字化数据，单位 kPa。
- `Budday_2017_brain_CX`：Budday 等（2017）的脑皮层拉伸、压缩与剪切数据，采用 Linka 等（2023）在 LivingMatterLab/CANN 中分发的版本，单位 kPa。原始工作簿及提取约定见 `brain-source.md`。

本仓库不再分发原论文图像、扫描件和工作簿。
原始实验的贡献归其作者所有；收录处理后的数值表不表示拥有底层数据，也不赋予新的数据许可证。
请保留上述引用并遵守来源条款。本地 CANN 源码许可证保留在 `../LICENSES/CANN-MIT.txt` 中，不视为适用于所有实验数据集的统一许可证。

`../results/figure_data/` 包含用于正文图像的固定版本数据表。
目标值、预测值列及合成数据记录保留了实际绘图数值。复现已发表图像时应使用这些表，而不应假定每份历史原始输入均采用最终流程。
合成数据生成器位于 `../src/nn_invariant/preprocess/synthetic_data.py`。
