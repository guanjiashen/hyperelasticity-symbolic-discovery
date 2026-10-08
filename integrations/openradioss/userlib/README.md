# LAW291 动态用户材料库

本目录通过 `/MAT/USER01` 实现 NN_invariant 基于不变量的超弹性本构方程，无须修改 Starter 或 Engine 源码。
Starter 例程仍可读取历史手工输入的 LAW291 系数，同时支持 NN_invariant 的 `PACKAGE` 和 `TRAIN` 材料卡。

实现针对 `Ismstr=0` 的实体单元，与 O-Ring 算例一致。USER01 接口未包含热膨胀。

构建命令：

```bash
./build.sh
```

生成的库为 `libraduser_law291.so`，关联验证算例为 `exec/cases/O-Ring_USER291_2026`。

将 NN_invariant 材料包转换为 OpenRadioss 扁平材料包：

```bash
python3 tools/nn_invariant/export_openradioss.py export-openradioss \
  --config /path/to/material_package.json \
  --output exec/cases/O-Ring_USER291_2026/nn_material.flat \
  --nu 0.499828708839903
```

在 `/MAT/USER01` 中使用已有扁平材料包：

```text
/MAT/USER01/2
Rubber_USER291_dynamic
2.0e-9
PACKAGE
nn_material.flat
0.499828708839903 1e30
2
```

`TRAIN` 使用相同布局，但需提供 JSON 材料包或配置路径及缓存目录：

```text
/MAT/USER01/2
Rubber_USER291_dynamic
2.0e-9
TRAIN
nn_train_config.json
nn_cache
0.499828708839903 1e30
2
```

设置后，Starter 会调用 `RAD_NN_PYTHON` 和 `RAD_NN_EXPORTER` 指定的解释器及导出器。
示例 `run.sh` 将 `RAD_NN_EXPORTER` 指向 OpenRadioss 部署封装。
Engine 例程仅使用数值 `UPARAM`，不依赖 Python、PyTorch、PySR、Julia 或 JSON。
NN_invariant 材料包导出为 `INVARIANT_POLYNOMIAL_TERMS_V1`（非负指数）或
`INVARIANT_LAURENT_TERMS_V1`（允许正负整数指数，如 `1/I2`），使 Engine 可计算超出旧 LAW291 固定系数模板的 PySR 表达式。
两种标签使用相同的逐行布局：

```text
NN_INVARIANT_MATERIAL_V1
INVARIANT_LAURENT_TERMS_V1
<nterms>
<coef> <p1> <p2>
<bulk_kappa>
<gref>
<I1_min> <I1_max>
<I2_min> <I2_max>
<rmse>
<model_hash>
```

系数行重复 `nterms` 次，每行对应 `W += coef * I1**p1 * I2**p2`。
`lecmuser01` 写入的 `UPARAM` 布局为：`uparam(1)=2`（格式），`uparam(2)=nterms`，随后为
`(coef, p1, p2)` 三元组及 `bulk_kappa, sigcut, iform, gref, nu, I1_min, I1_max, I2_min, I2_max, rmse`。
80 项的 `UPARAM` 容量最多支持 20 个表达式项。

## 与材料卡相关的训练选项

`TRAIN` 卡指定的 JSON 为 `tools/nn_invariant/README.md` 中描述的工作流配置。
最新 NN_invariant 快照新增的选项包括：

- `train.hidden_layers`：隐藏层宽度列表，如 `[16, 16, 8]`，用于多层网络；单层网络仍可用 `hidden_neurons`。
- `symbolic.simplify_expression` 及 `simplify_rmse_tolerance`、`simplify_energy_weight`、`simplify_stress_weight`：拟合后剪除加法项并重新拟合剩余系数。剪枝后的模型通常占用更少的 `UPARAM` 项，也可能包含负指数，因此提供 Laurent 格式标签。
- `symbolic.binary_operators` / `symbolic.unary_operators`：PySR 算子集合。请限制为 `+ - * /` 和 `square`/`cube`；除非设置 `allow_non_polynomial_operators`，否则导出器会拒绝其他算子。

Starter 输出清单会打印每个导出项的 `coef`、`I1^p1` 和 `I2^p2`，可在 `_0000.out` 中核对部署表达式。
