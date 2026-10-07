# NN 到符号表达式

本目录用于把 [ffbp_model.pt](/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/NN_output/ffbp_model.pt) 对应的基于不变量神经网络，转成更容易解释的符号应变能表达式。

流程分两步：

1. 在 $\lambda_1, \lambda_2 \in [1, 3]$ 网格内，用神经网络生成应变能数据。
2. 用 PySR 对这些数据进行符号回归，默认拟合 $W(I_1, I_2)$。

现在也支持第二种流程：

1. 用神经网络分别生成单轴拉伸 `UT`、纯剪切 `PS`、等双轴拉伸 `ET` 的数据。
2. 每条载荷路径都同时导出应变能和名义应力。
3. PySR 在搜索时同时考虑能量误差和名义应力误差，最终仍输出应变能表达式 $W$。

## 脚本

- `nn_to_symbolic.py export`
  - 加载训练好的神经网络。
  - 当 `--sampling-mode grid` 时，生成 `lambda1, lambda2, lambda3, I1, I2, energy, stress_lambda1, stress_lambda2` 数据表。
  - 当 `--sampling-mode loadcases` 时，生成 `UT/PS/ET` 路径上的 `stretch, lambda1, lambda2, lambda3, I1, I2, energy, nominal_stress` 数据表。
- `nn_to_symbolic.py fit`
  - 读取导出的 CSV。
  - 运行 PySR，输出最优符号表达式、能量预测结果，以及对神经网络训练所用实验数据的应力预测。
- `nn_to_symbolic.py pipeline`
  - 在同一个 Python 环境里连续执行 `export + fit`。
- `nn_to_symbolic.py predict-experiment`
  - 读取已有的符号表达式。
  - 在神经网络训练所用实验数据上回推应力预测并输出对比结果。

## 导出能量数据

如果当前环境已经有 `torch`，可以直接执行：

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python \
  /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  export \
  --lambda1-min 1.0 --lambda1-max 3.0 --lambda1-points 81 \
  --lambda2-min 1.0 --lambda2-max 3.0 --lambda2-points 81
```

默认输出：

- `../output/SR_output/nn_energy_grid.csv`
- `../output/SR_output/nn_energy_grid_metadata.json`

默认会减去未变形点 $W(1,1)$，使导出的能量以参考构型为零点。

如果你希望直接用神经网络生成 `UT/PS/ET` 三种载荷路径的数据，可以执行：

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python \
  /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  export \
  --sampling-mode loadcases \
  --output-csv /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output/nn_loadcase_samples.csv \
  --ut-stretch-range 1.0 8.0 --ut-points 80 \
  --ps-stretch-range 1.0 8.0 --ps-points 80 \
  --et-stretch-range 1.0 8.0 --et-points 80
```

这里三组拉伸区间都可以直接通过命令行输入。

## 进行 PySR 拟合

如果当前环境已经安装 `pysr`，执行：

```bash
python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  fit \
  --input-csv /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output/nn_energy_grid.csv \
  --feature-space invariants \
  --binary-operators + - '*' / \
  --stress-loss-weight 1.0 \
  --niterations 60 \
  --population-size 50 \
  --populations 10 \
  --maxsize 18 \
  --deterministic
```

推荐优先拟合 `--feature-space invariants`，得到 $W(I_1, I_2)$，这与当前神经网络的输入空间一致，也更符合超弹性本构表达。

二元算子可通过 `--binary-operators` 配置；实验回归使用 `+ - * /`，因此允许搜索包含有理项的应变能。除法由 PySR 的保护机制在搜索采样点上数值评估，但最终表达式仍需检查分母零点及加载路径上的渐近行为。

当 `--stress-loss-weight > 0` 时，PySR 在搜索阶段会使用联合目标：

$$
  ext{loss} = \text{MSE}_{energy} + w_{stress} \cdot \text{MSE}_{stress}
$$

其中能量项来自神经网络导出的应变能网格，
应力项来自同一 `lambda1-lambda2` 网格上、由神经网络应变能通过求导得到的应力误差，
而最终搜索得到的表达式本身仍然是应变能函数 $W$。

如果导出数据使用 `--sampling-mode loadcases`，那么应力项改为 `UT/PS/ET` 三条路径上的名义应力误差；
这时每个采样点的目标名义应力同样由神经网络应变能求导得到，而不是直接使用实验应力作为搜索目标。

当 `--feature-space invariants` 时，符号应力通过 $\partial W/\partial I_1$ 与 $\partial W/\partial I_2$ 再结合链式法则得到；
当 `--feature-space stretches` 时，PySR 拟合 $W(\lambda_1,\lambda_2,\lambda_3)$；应力由三个主拉伸导数沿不可压缩 UT、PS、ET 路径的链式法则重建。

附加 `--valanis-landel` 可将主拉伸回归限制为 $W=w(\lambda_1)+w(\lambda_2)+w(\lambda_3)$。该选项当前要求 `--sampling-mode loadcases`，并与 `--feature-space stretches` 一起使用。

拟合完成后，脚本会先得到 PySR 的原始表达式 $W_{raw}$，再自动做一次参考态归一化：

$$
W_{final} = W_{raw} - W_{raw}(I_1=3, I_2=3)
$$

因此最终写入 `best_equation.txt`、`predictions.csv`、`fit_summary.json` 和实验数据预测文件的，都是满足初始态零能量约束的最终表达式 `W_final`，而不是未修正的原始表达式。

### 拟合后自动简化

使用 `--simplify-expression` 可在 PySR 完成后逐项简化不变量表达式。程序会展开加法项，依次尝试删除一项，并用归一化的 NN 能量与实验应力联合最小二乘重新拟合剩余系数。只有当重拟合模型的选择分数相对原始表达式的增幅不超过 `--simplify-rmse-tolerance` 时，才接受删除。

相关参数为：

- `--simplify-rmse-tolerance`：允许的相对精度损失，默认为 `0.02`；
- `--simplify-energy-weight`：系数重拟合中的 NN 能量权重，默认为 `0.0`；
- `--simplify-stress-weight`：系数重拟合中的实验应力权重，默认为 `1.0`。

当前自动剪枝支持对系数呈线性的加法型 $W(I_1,I_2)$ 表达式。输出的 `raw_best_equation.txt` 保存简化前结果，`simplification_summary.json` 保存各项贡献、删除记录及精度—复杂度候选。接受简化后，`best_equation.txt`、预测结果和材料包均使用简化并重拟合后的表达式。

输出目录默认是：

- `../output/SR_output/best_equation.txt`
- `../output/SR_output/predictions.csv`
- `../output/SR_output/fit_summary.json`
- `../output/SR_output/energy_comparison_contours.png`
- `../output/SR_output/experiment_predictions.csv`
- `../output/SR_output/experiment_predictions.png`
- `../output/SR_output/experiment_prediction_summary.json`

其中 `best_equation.txt` 和 `fit_summary.json` 现在会额外记录：

- `raw_initial_state_energy`：原始 PySR 表达式在参考态的能量值。
- `initial_state_energy`：归一化后最终表达式在参考态的能量值，理论上应为 0。
- `initial_state_energy_is_zero`：最终表达式是否通过零参考能量检查。
- `stress_loss_weight`：搜索阶段应力损失项的权重。
- `experiment_stress_loss`：最终符号表达式在实验应力数据上的总体 MSE。
- `experiment_stress_rmse`：最终符号表达式在实验应力数据上的总体 RMSE。

其中 `energy_comparison_contours.png` 包含三幅图：

- 神经网络生成的应变能等高线
- PySR 符号表达式预测的应变能等高线
- 两者绝对误差等高线

其中实验数据预测相关文件含义如下：

- `experiment_predictions.csv`：按模式输出实验点的 stretch、应力真值、符号模型预测值，以及对应的 $I_1, I_2, \partial W/\partial I_1, \partial W/\partial I_2$。
- `experiment_predictions.png`：实验应力散点与符号模型应力曲线的对比图。
- `experiment_prediction_summary.json`：每个模式的 RMSE 和 $R^2$。

注意：由于最终发布的表达式会自动减去参考态常数项，最终 `rmse` 和 $R^2$ 是基于归一化后的 `W_final` 重新计算的，所以它们可能和未修正原始表达式的指标略有差异。

## 一步执行

如果同一个环境同时有 `torch` 和 `pysr`，可以直接运行：

```bash
python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py pipeline
```

`pipeline` 和 `fit` 在完成符号拟合后，会默认继续生成实验数据预测结果。

如果只想基于现有表达式重新生成实验数据预测，可以单独执行：

```bash
python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/Symbolic_Regression/nn_to_symbolic.py \
  predict-experiment \
  --equation-file /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output/best_equation.txt \
  --output-dir /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/output/SR_output
```

## 环境说明

当前仓库里，训练神经网络通常使用 [FFNN/.venv](/home/guanjs/NN-constitutive/FFNN/.venv)，而 PySR 相关环境可参考已有目录 [SymbolicRegression/PySR-master](/home/guanjs/NN-constitutive/SymbolicRegression/PySR-master)。如果 `torch` 和 `pysr` 不在同一环境，直接按 `export -> fit` 两步运行即可。

如果你启用了 `--stress-loss-weight > 0`，Julia 环境里还需要安装 `Zygote.jl`，因为搜索阶段会对候选应变能表达式求导来构造应力损失。
