# NN_invariant 与 OpenRadioss 的集成

本目录包含供 Starter 调用的部署封装，用于 NN_invariant 基于不变量的超弹性模型工作流。

OpenRadioss 侧调用方式：

```bash
python3 tools/nn_invariant/export_openradioss.py export-openradioss \
  --config nn_openradioss_config.json \
  --output nn_cache/material.flat \
  --nu 0.495
```

`--config` 可以是已有的 NN_invariant `material_package.json`，也可以是
`examples/nn_openradioss_config.json` 这样的 OpenRadioss 工作流配置。

封装输出扁平的 `NN_INVARIANT_MATERIAL_V1` 文件，每一项保存系数及 `I1`、`I2` 的整数指数：

- `INVARIANT_POLYNOMIAL_TERMS_V1`：所有指数均非负。
- `INVARIANT_LAURENT_TERMS_V1`：布局相同，但允许负指数，例如 PySR 使用 `/` 或拟合后剪枝保留的 `-1.688/I2`。由于等容不变量满足 `I1, I2 >= 3`，这些负幂项有定义。

导出器自动选择格式标签。Starter 将各项读入 `UPARAM`；Engine 对数值项计算导数，不导入 Python、PyTorch、PySR、Julia 或 JSON。
无法写为 `coef * I1**p * I2**q` 之和的表达式（如 `1/(I1+I2)`、`exp(I1)`）会在导出时被拒绝并报告明确错误。

## 工作流配置选项

`train` 配置块对应 `main.py` 参数：

| 配置项 | 命令行参数 | 说明 |
| --- | --- | --- |
| `hidden_neurons` | `--hidden-neurons` | 单隐藏层宽度，兼容历史配置 |
| `hidden_layers` | `--hidden-layers N1 N2 ...` | 各隐藏层宽度，如 `[16, 16, 8]`；覆盖 `hidden_neurons` |
| `activation`, `optimizer`, `learning_rate`, `epochs`, `print_every`, `lbfgs_max_iter`, `lbfgs_history_size`, `random_seed` | 同名参数 | |
| `synthetic_model`, `synthetic_point_count`, `synthetic_noise_std`, `synthetic_*_range` | 同名参数 | 仅用于 `"workflow": "synthetic"` |
| `mooney_rivlin_c10/c01`, `ogden_mu/alpha`, `arruda_boyce_mu`, `arruda_boyce_lambda_m` | 同名参数 | 合成模型参数 |

`symbolic` 配置块对应 `Symbolic_Regression/nn_to_symbolic.py pipeline`：

| 配置项 | 命令行参数 | 默认值 |
| --- | --- | --- |
| `binary_operators` | `--binary-operators` | `["+", "-", "*"]` |
| `unary_operators` | `--unary-operators` | `["square"]` |
| `simplify_expression` | `--simplify-expression` | `false` |
| `simplify_rmse_tolerance` | `--simplify-rmse-tolerance` | `0.02` |
| `simplify_energy_weight` | `--simplify-energy-weight` | `0.0` |
| `simplify_stress_weight` | `--simplify-stress-weight` | `1.0` |
| `niterations`, `population_size`, `populations`, `maxsize`, `seed`, `deterministic`, `energy_loss_weight`, `stress_loss_weight`, `bulk_kappa`, `*_points`, `*_stretch_range`, `singularity_threshold`, `fit_python`, `export_python`, `valanis_landel`, `skip_experiment_prediction` | 同名参数 | |

`simplify_expression` 启用拟合后剪枝：贪心删除最佳 PySR 表达式的加法项，重新拟合剩余系数，并在选择分数增幅不超过 `simplify_rmse_tolerance` 时接受简化模型。
结果写入 `SR_output/simplification_summary.json`，原始表达式保存于 `raw_best_equation.txt`，导出器打印删除的项。

默认只接受 `+ - * /` 和 `square`/`cube`，因为 Engine 求值器仅处理整数幂单项式。
设置 `"allow_non_polynomial_operators": true` 可向 PySR 传入其他算子；若最终表达式不能分解为支持的形式，导出仍会失败。

导出器写出扁平文件后会打印部署摘要，包括网络结构、剪枝结果、导出项及剪切/体积模量。
`doctor --config ...` 可报告解析后的工作流选项。

## 运行环境

- `PACKAGE` 模式只需 OpenRadioss 和用户材料共享库。
- `TRAIN` 模式需要 Python 及 `requirements.txt` 中的依赖，PySR 还需要可用的 Julia 安装。
- `RAD_NN_PYTHON` 指定 Starter 使用的 Python 解释器。
- `RAD_NN_EXPORTER` 可覆盖导出器脚本路径。
- 工作流配置可包含 `environment` 对象，建议在此设置 `PYTHON_JULIAPKG_EXE`、`PYTHON_JULIAPKG_PROJECT`、`JULIA_DEPOT_PATH` 等 PySR/Julia 环境变量，使 Julia 写入算例缓存而非只读虚拟环境。
- 默认复用已有 `material.flat` 和 `material_package.json`；设置 `force_retrain: true` 可强制重新生成。

## 同步内附源码

内附源码快照位于 `vendor/NN_invariant`，不包含虚拟环境、缓存、生成的图像、模型检查点和训练数据。
重新从上游 NN_invariant 同步时，须保留 OpenRadioss 专用补丁：`main.py` 和
`Symbolic_Regression/modules/config.py` 读取 `NN_INVARIANT_OUTPUT_DIR`，使输出进入算例缓存而非 `vendor/NN_invariant/output`。

```bash
rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.venv/' --exclude '.trash/' \
  --exclude 'output/' --exclude 'FEM/' --exclude 'synthetic_data/' \
  --exclude 'notes.md' --exclude 'SR_notes.md' --exclude 'run_fem.sh' \
  --exclude 'copy_output_to_windows.sh' \
  /path/to/NN_invariant/ tools/nn_invariant/vendor/NN_invariant/
# 随后重新应用 main.py 和 modules/config.py 中的 NN_INVARIANT_OUTPUT_DIR 补丁
```
