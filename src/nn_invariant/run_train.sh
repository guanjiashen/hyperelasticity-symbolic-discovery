#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../../.." && pwd)"
python_bin="$workspace_root/FFNN/.venv/bin/python"
train_script="$script_dir/main.py"

# =============================================================================
# Neural-network training configuration / 神经网络训练配置
# Command-line arguments appended to this script override these defaults.
# 追加到脚本后的同名命令行参数会覆盖这里的默认值。
# =============================================================================

# Neural-network hyperparameters / 神经网络超参数
# train_mode:
#   stress -> train from nominal-stress data / 使用名义应力训练
#   energy -> train from strain-energy data / 使用应变能训练
train_mode="stress"            # energy | stress
# Train every listed width. Multiple widths are written to NN_output-<width>/.
# 依次训练数组中的宽度；多个宽度分别写入 NN_output-<宽度>/。
hidden_neurons=(15 30 45 60)
activation="softplus"          # sigmoid | relu | tanh | gelu | selu | leaky_relu | elu | softplus
optimizer="lbfgs"              # adam | lbfgs
learning_rate="0.2"            # Optimizer learning rate / 优化器学习率
epochs="200"                   # Number of outer training epochs / 外层训练轮数
print_every="20"               # Print losses every N epochs / 每 N 轮输出一次损失
lbfgs_max_iter="30"            # LBFGS inner iterations; ignored by Adam / LBFGS 内层迭代数
lbfgs_history_size="50"        # LBFGS history; ignored by Adam / LBFGS 历史长度
random_seed="42"               # NumPy/PyTorch seed / 随机种子

# Data workflow / 数据模式: synthetic | experimental
# synthetic generates analytical-model data; experimental reads the files below.
# synthetic 生成解析模型数据；experimental 读取下方文件。
workflow_mode="experimental"

# Synthetic constitutive model / 合成本构模型:
# ogden | arruda_boyce | mooney_rivlin
# Only used when workflow_mode="synthetic".
synthetic_model="ogden"

# Multi-term Ogden parameters. The two arrays must have equal lengths.
# 多项 Ogden 参数，mu 与 alpha 数组长度必须一致。
# Example / 示例: ogden_mu=(0.45 0.08), ogden_alpha=(2.2 -2.0)
ogden_mu=(0.45)
ogden_alpha=(2.2)

# Used only by synthetic_model="mooney_rivlin" / 仅用于 Mooney–Rivlin 合成模型
mooney_rivlin_c10="0.18"
mooney_rivlin_c01="0.02"

# Used only by synthetic_model="arruda_boyce" / 仅用于 Arruda–Boyce 合成模型
arruda_boyce_mu="0.24"
arruda_boyce_lambda_m="3.5"

# Synthetic stretch ranges for UT, PS, and ET / 合成 UT、PS、ET 伸长范围
synthetic_ut_min="1.0"
synthetic_ut_max="3.0"
synthetic_ps_min="1.0"
synthetic_ps_max="3.0"
synthetic_et_min="1.0"
synthetic_et_max="3.0"
synthetic_point_count="40"     # Samples per loading mode / 每种加载模式的样本数
synthetic_noise_std="0.0"      # Gaussian stress-noise standard deviation / 应力高斯噪声标准差

# Experimental datasets: column 1 nominal stress, column 2 stretch.
# 实验数据：第 1 列为名义应力，第 2 列为伸长比。
ut_dataset="$workspace_root/FFNN/fitting-data-PK/Treloar_1944/UT/stress_stretch.txt"
ps_dataset="$workspace_root/FFNN/fitting-data-PK/Treloar_1944/PS/stress_stretch.txt"
et_dataset="$workspace_root/FFNN/fitting-data-PK/Treloar_1944/ET/stress_stretch.txt"

extra_train_args=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--workflow)
			if [[ $# -lt 2 ]]; then
				echo "Missing value after --workflow" >&2
				exit 1
			fi
			workflow_mode="$2"
			shift 2
			;;
		--synthetic)
			workflow_mode="synthetic"
			shift
			;;
		--experimental)
			workflow_mode="experimental"
			shift
			;;
		--hidden-neurons)
			hidden_neurons=("$2")
			shift 2
			;;
		*)
			extra_train_args+=("$1")
			shift
			;;
	esac
done

if [[ "$workflow_mode" != "synthetic" && "$workflow_mode" != "experimental" ]]; then
	echo "Unsupported workflow: $workflow_mode. Expected synthetic or experimental." >&2
	exit 1
fi

if [[ ! -x "$python_bin" ]]; then
	echo "Python interpreter not found: $python_bin" >&2
	exit 1
fi

train_args=(
	# The options below are consumed by main.py.
	# 以下选项由 main.py 读取。
	--train-mode "$train_mode"
	--activation "$activation"
	--optimizer "$optimizer"
	--learning-rate "$learning_rate"
	--epochs "$epochs"
	--print-every "$print_every"
	--lbfgs-max-iter "$lbfgs_max_iter"
	--lbfgs-history-size "$lbfgs_history_size"
	--random-seed "$random_seed"
)

if [[ "$workflow_mode" == "synthetic" ]]; then
	# Generate UT/PS/ET data and save it under synthetic_data/<model>/.
	# 生成 UT/PS/ET 数据，并保存到 synthetic_data/<model>/。
	train_args+=(
		--use-synthetic-data
		--synthetic-model "$synthetic_model"
		--ogden-mu "${ogden_mu[@]}"
		--ogden-alpha "${ogden_alpha[@]}"
		--synthetic-point-count "$synthetic_point_count"
		--synthetic-noise-std "$synthetic_noise_std"
		--synthetic-ut-range "$synthetic_ut_min" "$synthetic_ut_max"
		--synthetic-ps-range "$synthetic_ps_min" "$synthetic_ps_max"
		--synthetic-et-range "$synthetic_et_min" "$synthetic_et_max"
		--mooney-rivlin-c10 "$mooney_rivlin_c10"
		--mooney-rivlin-c01 "$mooney_rivlin_c01"
		--arruda-boyce-mu "$arruda_boyce_mu"
		--arruda-boyce-lambda-m "$arruda_boyce_lambda_m"
	)
else
	# Train directly from the configured experimental files.
	# 直接读取配置的实验数据训练。
	train_args+=(
		--ut-dataset "$ut_dataset"
		--ps-dataset "$ps_dataset"
		--et-dataset "$et_dataset"
	)
fi

echo "Training neural network"
echo "Workflow mode: $workflow_mode"
for hidden_width in "${hidden_neurons[@]}"; do
	model_args=(--hidden-neurons "$hidden_width")
	if [[ ${#hidden_neurons[@]} -gt 1 ]]; then
		model_args+=(--output-suffix "$hidden_width")
	fi
	echo "Hidden neurons: $hidden_width"
	"$python_bin" "$train_script" "${train_args[@]}" "${model_args[@]}" "${extra_train_args[@]}"
done
