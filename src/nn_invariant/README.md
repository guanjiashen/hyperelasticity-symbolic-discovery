# NN_invariant 使用说明

本目录用于训练基于不变量的超弹性神经网络模型，支持：

- 实验数据训练（UT/PS/ET）
- 可选双轴数据（BS 或 BL）
- 可选解析模型合成数据（Ogden、Arruda-Boyce 或 Mooney-Rivlin）

## 1. 运行方式

在仓库根目录或当前目录执行均可。推荐使用项目虚拟环境 Python：

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py [命令行参数]

最简单运行（默认 Treloar UT/PS/ET，stress 模式）：

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py

默认输出目录：

- `output/NN_output`：神经网络训练结果，如 `ffbp_model.pt`、`ffbp_weights.txt`、应力预测图、能量等高线图、loss 曲线
- `output/SR_output`：符号回归结果，如导出的能量 CSV、PySR 拟合结果、能量对比等高线图

## 2. 命令行参数

参数定义位置：cli/cli.py

### 训练目标

- --train-mode {energy,stress}
  - 训练目标类型
  - 默认：stress

### 神经网络结构

- `--hidden-neurons N`
  - 单隐藏层宽度，默认：3
  - 保留用于兼容原有训练命令和模型

- `--hidden-layers N1 N2 ...`
  - 指定一个或多个隐藏层的宽度，并覆盖 `--hidden-neurons`
  - 例如 `--hidden-layers 16 16 8` 表示网络结构 `2 → 16 → 16 → 8 → 1`

多层网络训练示例：

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python \
  /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py \
  --hidden-layers 16 16 8 \
  --activation softplus
```

### 合成数据相关

- --use-synthetic-data
  - 启用后，不使用实验数据路径，而是按解析模型自动生成数据
  - 默认：关闭（不传该参数）

- --synthetic-model {ogden,arruda_boyce,mooney_rivlin}
  - 合成数据使用的解析模型
  - 默认：ogden

- --mooney-rivlin-c10 C10
  - 当 `--synthetic-model mooney_rivlin` 时使用的 Mooney-Rivlin `C10`
  - 默认：0.18

- --mooney-rivlin-c01 C01
  - 当 `--synthetic-model mooney_rivlin` 时使用的 Mooney-Rivlin `C01`
  - 默认：0.02

- --synthetic-point-count N
  - 每个模式采样点数量
  - 默认：30

- --synthetic-noise-std STD
  - 施加到合成应力数据上的高斯噪声标准差
  - 默认：0.0

- --synthetic-ut-range MIN MAX
  - UT 拉伸范围
  - 默认：1.0 3.0

- --synthetic-ps-range MIN MAX
  - PS 拉伸范围
  - 默认：1.0 3.0

- --synthetic-et-range MIN MAX
  - ET 拉伸范围
  - 默认：1.0 3.0

### 实验数据路径（UT/PS/ET）

- --ut-dataset 路径
- --ps-dataset 路径
- --et-dataset 路径

默认分别指向：

- fitting-data-PK/Treloar_1944/UT/stress_stretch.txt
- fitting-data-PK/Treloar_1944/PS/stress_stretch.txt
- fitting-data-PK/Treloar_1944/ET/stress_stretch.txt

### 双轴数据相关

- --enable-biaxial
  - 启用双轴数据读取
  - 默认：关闭（不传该参数）

- --biaxial-prefix {BS,BL}
  - 双轴模式前缀
  - BS 通常对应 BT_small
  - BL 通常对应 BT_large
  - 默认：BS

- --biaxial-base-dir 路径
  - 双轴数据根目录，目录下应包含 B1、B2... 子目录，每个子目录内有 stress_stretch.txt
  - 默认：fitting-data-PK/Kawabata_1981/BT_small

说明：双轴数据数量不是写死的，程序会自动扫描你给的目录中所有满足条件的 B* 子目录。

## 3. 示例命令

### 示例 0：依次执行训练和符号回归

训练和符号回归现在由两个独立脚本负责。默认使用合成数据工作流，依次运行：

```bash
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_train.sh
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_symbolic.sh
```

默认行为等价于：

- 训练：`main.py --train-mode stress --use-synthetic-data --synthetic-model mooney_rivlin`
- 符号回归：`nn_to_symbolic.py pipeline`

`run_train.sh` 默认先用 Mooney-Rivlin 模型生成 `UT/PS/ET` 数据并训练 `NN_invariant`。训练完成后，`run_symbolic.sh` 对保存的网络做符号回归，并默认执行 FEM 单元测试。两个脚本必须选择相同的工作流。

如果想切回实验数据工作流，可以显式传入：

```bash
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_train.sh --experimental
bash /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_symbolic.sh --experimental
```

也可以显式指定：

- `--synthetic`：使用当前默认的合成数据工作流
- `--experimental`：使用 Treloar `UT/PS/ET` 实验数据工作流
- `--run-fem`：执行末尾 FEM 单元测试（仅适用于 `run_symbolic.sh`，也是默认行为）
- `--skip-fem`：跳过末尾 FEM 单元测试（仅适用于 `run_symbolic.sh`）

如果你想改训练参数或符号回归参数，分别编辑 [run_train.sh](/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_train.sh) 和 [run_symbolic.sh](/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/run_symbolic.sh) 中的参数数组：

- `train_args=(...)`
- `symbolic_args=(...)`

例如可以把它们改成：

```bash
train_args=(
  --train-mode stress
)

symbolic_args=(
  pipeline
  --lambda1-points 41
  --lambda2-points 41
  --niterations 20
  --deterministic
)
```

### 示例 A：只用默认实验数据训练（stress）

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress

### 示例 B：启用双轴小变形数据（BT_small -> BS）

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --enable-biaxial --biaxial-prefix BS --biaxial-base-dir /home/guanjs/NN-constitutive/FFNN/fitting-data-PK/Kawabata_1981/BT_small

### 示例 C：启用双轴大变形数据（BT_large -> BL）

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --enable-biaxial --biaxial-prefix BL --biaxial-base-dir /home/guanjs/NN-constitutive/FFNN/fitting-data-PK/Kawabata_1981/BT_large

### 示例 D：使用合成数据（Ogden）

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --use-synthetic-data --synthetic-model ogden --synthetic-point-count 40 --synthetic-ut-range 1.0 2.5 --synthetic-ps-range 1.0 2.5 --synthetic-et-range 1.0 2.5

### 示例 E：使用合成数据（Mooney-Rivlin）

/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --use-synthetic-data --synthetic-model mooney_rivlin --mooney-rivlin-c10 0.18 --mooney-rivlin-c01 0.02 --synthetic-point-count 40 --synthetic-ut-range 1.0 3.0 --synthetic-ps-range 1.0 3.0 --synthetic-et-range 1.0 3.0

如果想在解析模型生成的数据中加入可控噪声，例如标准差为 `0.02` 的高斯噪声，可以追加：

```bash
/home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --train-mode stress --use-synthetic-data --synthetic-model mooney_rivlin --mooney-rivlin-c10 0.18 --mooney-rivlin-c01 0.02 --synthetic-point-count 40 --synthetic-noise-std 0.02 --synthetic-ut-range 1.0 3.0 --synthetic-ps-range 1.0 3.0 --synthetic-et-range 1.0 3.0
```

## 4. 常见问题

1. 报错找不到数据文件

- 检查 --ut-dataset / --ps-dataset / --et-dataset 路径是否存在。
- 启用双轴时，检查 --biaxial-base-dir 下是否有 B* 子目录且每个目录内有 stress_stretch.txt。

1. 想确认参数是否生效

- 可先执行：
  /home/guanjs/NN-constitutive/FFNN/.venv/bin/python /home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant/main.py --help
