# figure10_cann_csr：在 OpenRadioss Starter 中使用 CANN–CSR 流程重算第 4.4 节算例

使用的仓库为 `/home/guanjs/OpenRadioss-latest-20251211`（WSL），材料卡为 TRAIN 模式的 `/MAT/USER01`（LAW291）。
此前未使用 CANN–CSR 的结果归档在 `../../backup/20260918_sec44_eq48/`。
本文件保留历史算例记录，其中表图编号及部分路径对应原工作目录。

## 计算流程

1. Starter（`FIG10_CANN_CSR_0000.rad`，配置 `cann_csr_config.json`）：
   - CANN：8 个 Softplus 神经元，L-BFGS 学习率 0.2，400 轮训练、每轮 30 次内部迭代，随机种子 42，应力训练模式，训练/验证/测试划分 70/15/15。47 个样本、33 个参数，耗时 3.8 s；三组应力 MSE 分别为 1.26e-4 / 6.8e-5 / 1.0e-4 MPa²。
   - CSR（PySR）：算子 `+ - * square`，不使用除法。此前包含除法的搜索得到分母为 `I1^2 - I2^3` 的有理模型，导出时被拒绝，见 `attempt1_div/`。12 个种群、每个 60 个个体，180 次迭代，最大表达式大小 16，能量/应力权重 0.1/1.0，每条路径 32 个点，剪枝容差 2%，未删除任何项。
   - 原始候选表达式：`(I1 - 7.7473)(2.5884e-3 I1^2 - 6.7919e-3 I2 + 0.20517)`。
   - 导出表达式满足 `Psi(3,3)=0`：
     `Psi = 0.20517 I1 + 5.2619e-2 I2 - 2.0053e-2 I1^2 - 6.7919e-3 I1 I2 + 2.5884e-3 I1^3 - 0.60164 [MPa]`。
     `G0 = 0.3332 MPa`，`kappa = 33.21 MPa`，`nu = 0.495`；`nn_cache/material.flat` 的模型哈希为 `5c4133ef4ad76fa7`。
   - 47 个实验点的应力 MSE：总体 3.6990e-4；UT 2.3134e-4，PS 5.1351e-4，ET 3.6329e-4 MPa²。
2. Engine（`FIG10_CANN_CSR_0001.rad`）：1,717,697 个计算循环，终止时刻 0.18 s，正常结束（NORMAL TERMINATION），耗时 823.6 s，能量误差约 -0.4%，无附加质量。
3. 后处理（`postprocess.py`、`select_state.py`、`plot_identification.py`）：选取时刻 0.1325 s（第 54 帧），反力 19.98 N，夹持端位移 54.39 mm，KE/IE 为 0.041%，单元最大 lambda1 为 2.60；路径 lambda1 为 1.12–2.42，最大值位于 s = 57.02 mm。输出为 `selected_state.json`、`qa_summary.json`、`path_maximum_stretch.csv` 和 `figs/heterogeneous_cann_csr_{fit,results,path}.{png,pdf}`。

## 运行方式（WSL）

见 `run_cann_csr_wsl.sh` 和 `../../tmp/launch_stage.sh`，阶段为 `starter | engine | post`。

## 网格敏感性重算（加密网格）

算例目录为 `../figure10_cann_csr_fine/`，脚本 `run_fine_wsl.sh` 支持 `starter | engine | post | all`，日志为 `fine_run.log`。
使用同一 `material.flat` 材料包（哈希 `5c4133ef4ad76fa7`，PACKAGE 模式）、边界条件及 65 mm / 0.18 s 位移加载。
Gmsh 平面网格尺寸为 0.75–1.50 mm（粗网格为 1.20–2.50 mm），厚度方向两层棱柱单元，共 6,387 个节点、7,706 个 PENTA6 单元。
Engine 经 2,257,221 个循环正常结束，耗时 2271.8 s，能量误差约 -0.3%，无附加质量。

选取时刻 0.1375 s：反力 19.92 N，夹持端位移 56.39 mm，KE/IE 为 0.033%，单元最大 lambda1 为 2.82；路径 lambda1 为 1.10–2.48，最大值位于 s = 57.02 mm。
粗、细网格路径差的最大值为 0.095，RMS 为 0.041（2.2%）；反力差不超过 0.85 N，在 u = 54.39 mm 时为 0.82 N（4.1%）；相对实验的路径 RMSE 分别为 0.047 和 0.063。
`compare_meshes.py` 生成 `figs/heterogeneous_cann_csr_{mesh,path_mesh}_comparison.{png,pdf}` 和 `../figure10_cann_csr_fine/mesh_comparison_summary.json`。
历史论文编号中，原网格称为粗网格（图 17(b)、表 7），细网格称为加密网格；图 17(c) 使用共享色标（`compare_meshes.py`）。

## 本构求值耗时基准（历史表 8）

`benchmark_evaluation_cost.py` 在 WSL 的 NN 虚拟环境中，对比训练好的 8 神经元 Softplus CANN
（`nn_cache/output/NN_output/ffbp_model.pt`，33 个参数）与导出的符号模型（`material_package.json`）计算 `(dPsi/dI1, dPsi/dI2)` 的耗时。
使用单线程、float64 和 Intel Core Ultra 5 235。
两种模式使用同一组 1e6 个不变量样本：向量化模式一次调用处理全部点；逐点模式每点调用一次并丢弃结果，使三个求值器使用同一循环框架。
Python 计时在不同进程间可能波动数十个百分点，因此采用三次运行的中位数：
`aggregate_evaluation_cost.py evaluation_cost_run{1,2,3}.json` → `evaluation_cost.json`。

中位耗时：向量化 CANN 自动微分 129.9 ns，CANN 闭式 NumPy 96.0 ns，SR 6.0 ns（相对自动微分快 21.5 倍）；
逐点 CANN 自动微分 36.71 μs，CANN 闭式标量 0.967 μs，SR 0.086 μs（分别快 425 倍和 11 倍）。历史表 8 仅列出 PyTorch 自动微分和符号模型。
单元求值次数：粗网格 `2,962 × 1,717,697 ≈ 5.1e9`，细网格 `7,706 × 2,257,221 ≈ 1.7e10`。
按粗网格次数外推，单线程逐点自动微分约需 52 h，符号模型约需 7 min。

## Engine 内的本构耗时与稳健性（历史表 9，engine_cost/）

基准用户库均由 `build_cann_userlib.sh` 构建：

- `engine_cost/luser01.f90`：`make_cann_luser01.py` 从已部署的 `law291_userlib/luser01.f90` 生成，以闭式形式求值训练好的 8 神经元 Softplus CANN，权重来自 `nn_cache/output/NN_output/ffbp_model.pt`，生成 `libraduser_cann291.so`。
- `engine_cost/luser01_specialized.f90`：`make_specialized_luser01.py` 从同一例程生成，在构建时固定发现模型的指数，采用 Horner 形式，系数从 `nn_cache/material.flat` 读取，生成 `libraduser_spec291.so`。

两者仅替换返回 `(dPsi/dI1, dPsi/dI2)` 的调用。
可用 `smoke_test.sh <lib>` 验证任一库，计算至 0.002 s，预期正常结束并经历 18,460 个循环。

- `run_fortran_bench.sh` 与 `bench_constitutive.f90`：仅测试材料例程，2e7 个样本，`gfortran -O3`。专用多项式 0.79 ns/次，CANN 闭式求值 16.4 ns/次，导出项表求值 43.3 ns/次；CANN/专用多项式耗时比为 20.8。日志为 `bench_constitutive.log`。
- `run_engine_cost_repeat3.sh` 与 `collect_repeats3.py`（历史表 9）：从同一重启动文件和输入文件交错运行三种 Engine 实现，t ≤ 0.02 s，8 线程，各重复 5 次（`rep_table_*`、`rep_spec_*`、`rep_cann_*`）。两种符号例程均为 184,733 个循环，打印的能量一致；CANN 为 184,739 个循环。每次求值的单元块 CPU 耗时中位数分别为专用多项式 712.2、项表 736.0、CANN 749.1 ns，范围分别为 700–729、728–743、727–755 ns。总耗时中位数为 91.7 / 94.0 / 94.0 s，波动约 3%。CANN 与专用多项式的差为 36.9 ns/次，约占单元块时间的 5%，与运行间波动同量级；单个单元循环的 CPU 耗时约 712 ns，本构求值占比不超过 6%。输出为 `engine_cost_repeats3.json`。
- `run_engine_cost_repeat.sh` 与 `collect_repeats.py`：同一协议较早的两种实现对比（部署项表与 CANN），已由三种实现的运行替代。输出为 `engine_cost_repeats.json`。
- `run_engine_cost.sh`：单次配对运行。t ≤ 0.10 s（`run_symbolic/`、`run_cann/`）时，循环数为 936,924 / 936,353，耗时为 466.6 / 477.6 s，单元 CPU 时间为 2055 / 2111 s（`engine_cost.json`）。完整 0.18 s 加载（`run_symbolic_full/`、`run_cann_full/`）中，符号模型重现正式计算结果（1,717,697 个循环，844.1 s）；CANN 驱动计算在 t = 0.1188 s、1,118,751 个循环后失稳，表现为节点时间步崩溃、能量误差 -99.9%（`engine_cost_full.json`）。
- `extrapolation_check.py`：CANN 的单轴名义应力在 lambda = 2.48 达峰，超过约 2.7 后变为负值，刚好超出标定域（I1 ≤ 7.674，对应 lambda ≤ 2.63）；导出的多项式在 lambda = 3.5 以内保持单调，而韧带区域达到 lambda1 = 2.60–2.82。输出为 `extrapolation_check.json`。
