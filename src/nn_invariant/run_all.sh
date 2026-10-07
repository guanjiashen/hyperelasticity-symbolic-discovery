#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../../.." && pwd)"

# =============================================================================
# Shared workflow configuration. Edit common settings in this section only.
# 共享工作流配置：通常只需要修改这一段。
# =============================================================================

# Workflow options / 工作流选项:
#   experimental : use the UT/PS/ET files configured below（使用下方实验数据）
#   synthetic    : generate data from synthetic_model（使用解析模型生成数据）
workflow_mode="experimental"

# Synthetic model options / 合成模型选项: ogden | mooney-rivlin
# This setting is ignored when workflow_mode="experimental".
# 在实验数据模式下，此项不会参与训练或回归。
synthetic_model="ogden"

# Symbolic-regression variables / 符号回归变量:
#   invariants : fit W(I1, I2); Valanis-Landel must be false
#               回归不变量形式 W(I1, I2)，必须关闭 Valanis–Landel
#   stretches  : fit W(lambda1, lambda2, lambda3)
#               回归主拉伸形式 W(lambda1, lambda2, lambda3)
symbolic_feature_space="invariants"

# When true with stretches, constrain W to
# W = w(lambda1) + w(lambda2) + w(lambda3).
# stretches 模式下设为 true，可施加 Valanis–Landel 可分离假设。
use_valanis_landel=false

# Symbolic-regression loss weights / 符号回归损失权重:
# total_loss = energy_loss_weight * energy_MSE
#            + stress_loss_weight * stress_MSE
# A zero stress weight skips stress differentiation during candidate evaluation.
# 应力权重为 0 时，候选表达式评估会跳过应力求导。
energy_loss_weight="1.0"
stress_loss_weight="1.0"

# Post-fit SR simplification / SR 拟合后简化
simplify_expression=true
simplify_rmse_tolerance="0.02"
simplify_energy_weight="0.0"
simplify_stress_weight="1.0"

# Neural-network hyperparameters / 神经网络超参数
train_mode="stress"            # energy | stress
hidden_neurons="3"
activation="softplus"          # sigmoid | relu | tanh | gelu | selu | leaky_relu | elu | softplus
optimizer="lbfgs"              # adam | lbfgs
learning_rate="0.2"
epochs="200"
print_every="20"
lbfgs_max_iter="30"            # Used only with LBFGS
lbfgs_history_size="50"        # Used only with LBFGS
random_seed="42"

# Ogden parameters. Multiple terms are written as matching arrays, for example:
# 多项 Ogden 模型要求 mu 和 alpha 数量一致，例如：
#   ogden_mu=(0.45 0.08)
#   ogden_alpha=(2.2 -2.0)
ogden_mu=(0.45)
ogden_alpha=(2.2)

# Mooney-Rivlin parameters. Used only when synthetic_model="mooney-rivlin".
# 仅在选择 Mooney–Rivlin 合成模型时生效。
mooney_rivlin_c10="0.18"
mooney_rivlin_c01="0.02"

# Synthetic stretch ranges for uniaxial tension (UT), pure shear (PS), and
# equibiaxial tension (ET). These ranges are shared by training and symbolic
# sampling. They are ignored in experimental mode, where ranges come from data.
# 合成数据的 UT/PS/ET 伸长范围；实验模式会直接从数据文件读取范围。
ut_stretch_min="1.0"
ut_stretch_max="3.0"
ps_stretch_min="1.0"
ps_stretch_max="3.0"
et_stretch_min="1.0"
et_stretch_max="3.0"
synthetic_point_count="40"
synthetic_noise_std="0.0"

# Experimental stress-stretch datasets. Each file is expected to contain
# stress in column 1 and stretch in column 2.
# 实验应力-伸长数据：第 1 列为应力，第 2 列为伸长。
ut_dataset="$workspace_root/FFNN/fitting-data-PK/Treloar_1944/UT/stress_stretch.txt"
ps_dataset="$workspace_root/FFNN/fitting-data-PK/Treloar_1944/PS/stress_stretch.txt"
et_dataset="$workspace_root/FFNN/fitting-data-PK/Treloar_1944/ET/stress_stretch.txt"

# Stage switches. Set a value to false to disable that stage by default.
# 阶段开关：设为 false 可默认跳过对应阶段。
run_train=true
run_symbolic=true
run_fem=true

show_help() {
	cat <<'EOF'
Usage: ./run_all.sh [OPTIONS]

Run NN training, symbolic regression, and FEM sequentially with one shared
configuration block.

Options:
  --experimental       Use the configured UT/PS/ET experimental datasets.
  --synthetic          Generate and use synthetic constitutive-model data.
  --synthetic-model MODEL
					   Select ogden or mooney-rivlin for synthetic data.
  --synthetic-noise-std VALUE
                       Set Gaussian stress-noise standard deviation.
  --workflow MODE      Explicitly select experimental or synthetic.
  --feature-space SPACE
                       Select invariants or stretches for symbolic regression.
  --valanis-landel     Enable the Valanis-Landel stretch decomposition.
  --no-valanis-landel  Disable the Valanis-Landel stretch decomposition.
  --energy-loss-weight VALUE
                       Set the symbolic energy-loss weight.
  --stress-loss-weight VALUE
                       Set the symbolic stress-loss weight; 0 disables it.
  --simplify-expression
                       Enable post-fit term pruning and coefficient refitting.
  --no-simplify-expression
                       Disable post-fit expression simplification.
  --simplify-rmse-tolerance VALUE
                       Set the accepted relative simplification loss.
  --simplify-energy-weight VALUE
                       Set the normalized NN-energy refit weight.
  --simplify-stress-weight VALUE
                       Set the normalized experimental-stress refit weight.
  --skip-train         Skip neural-network training.
  --skip-symbolic      Skip symbolic regression.
  --skip-fem           Skip the FEM single-element test.
  -h, --help           Show this help message.

Examples:
  ./run_all.sh
  ./run_all.sh --synthetic
  ./run_all.sh --synthetic --synthetic-model mooney-rivlin
  ./run_all.sh --synthetic --synthetic-model mooney-rivlin --synthetic-noise-std 0.02
  ./run_all.sh --feature-space invariants --no-valanis-landel
  ./run_all.sh --feature-space stretches --valanis-landel
  ./run_all.sh --energy-loss-weight 1.0 --stress-loss-weight 1.0
  ./run_all.sh --experimental --skip-train
  ./run_all.sh --skip-fem
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--workflow)
			workflow_mode="$2"
			shift 2
			;;
		--synthetic)
			workflow_mode="synthetic"
			shift
			;;
		--synthetic-model)
			if [[ $# -lt 2 ]]; then
				echo "Missing value after --synthetic-model" >&2
				exit 1
			fi
			synthetic_model="$2"
			shift 2
			;;
		--synthetic-noise-std)
			if [[ $# -lt 2 ]]; then
				echo "Missing value after --synthetic-noise-std" >&2
				exit 1
			fi
			synthetic_noise_std="$2"
			shift 2
			;;
		--experimental)
			workflow_mode="experimental"
			shift
			;;
		--feature-space)
			symbolic_feature_space="$2"
			shift 2
			;;
		--valanis-landel)
			use_valanis_landel=true
			shift
			;;
		--no-valanis-landel)
			use_valanis_landel=false
			shift
			;;
		--energy-loss-weight)
			energy_loss_weight="$2"
			shift 2
			;;
		--stress-loss-weight)
			stress_loss_weight="$2"
			shift 2
			;;
		--simplify-expression)
			simplify_expression=true
			shift
			;;
		--no-simplify-expression)
			simplify_expression=false
			shift
			;;
		--simplify-rmse-tolerance)
			simplify_rmse_tolerance="$2"
			shift 2
			;;
		--simplify-energy-weight)
			simplify_energy_weight="$2"
			shift 2
			;;
		--simplify-stress-weight)
			simplify_stress_weight="$2"
			shift 2
			;;
		--skip-train)
			run_train=false
			shift
			;;
		--skip-symbolic)
			run_symbolic=false
			shift
			;;
		--skip-fem)
			run_fem=false
			shift
			;;
		-h|--help)
			show_help
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			exit 2
			;;
	esac
done

if [[ "$workflow_mode" != "synthetic" && "$workflow_mode" != "experimental" ]]; then
	echo "Unsupported workflow: $workflow_mode. Expected synthetic or experimental." >&2
	exit 1
fi

if [[ "$synthetic_model" == "mooney-rivlin" ]]; then
	synthetic_model="mooney_rivlin"
fi

if [[ "$synthetic_model" != "ogden" && "$synthetic_model" != "mooney_rivlin" ]]; then
	echo "Unsupported synthetic model: $synthetic_model. Expected ogden or mooney-rivlin." >&2
	exit 1
fi

if [[ "$symbolic_feature_space" != "invariants" && "$symbolic_feature_space" != "stretches" ]]; then
	echo "Unsupported feature space: $symbolic_feature_space. Expected invariants or stretches." >&2
	exit 1
fi

if [[ "$use_valanis_landel" == true && "$symbolic_feature_space" != "stretches" ]]; then
	echo "Valanis-Landel regression requires symbolic_feature_space=stretches." >&2
	exit 1
fi

common_args=(
	--workflow "$workflow_mode"
	--ut-dataset "$ut_dataset"
	--ps-dataset "$ps_dataset"
	--et-dataset "$et_dataset"
)

train_hyperparameter_args=(
	--train-mode "$train_mode"
	--hidden-neurons "$hidden_neurons"
	--activation "$activation"
	--optimizer "$optimizer"
	--learning-rate "$learning_rate"
	--epochs "$epochs"
	--print-every "$print_every"
	--lbfgs-max-iter "$lbfgs_max_iter"
	--lbfgs-history-size "$lbfgs_history_size"
	--random-seed "$random_seed"
)

symbolic_config_args=(
	--feature-space "$symbolic_feature_space"
	--energy-loss-weight "$energy_loss_weight"
	--stress-loss-weight "$stress_loss_weight"
	--simplify-rmse-tolerance "$simplify_rmse_tolerance"
	--simplify-energy-weight "$simplify_energy_weight"
	--simplify-stress-weight "$simplify_stress_weight"
)
if [[ "$simplify_expression" == true ]]; then
	symbolic_config_args+=(--simplify-expression)
else
	symbolic_config_args+=(--no-simplify-expression)
fi
if [[ "$use_valanis_landel" == true ]]; then
	symbolic_config_args+=(--valanis-landel)
else
	symbolic_config_args+=(--no-valanis-landel)
fi

# Synthetic-only arguments are forwarded to the stages that consume them.
# 合成模型参数只传给需要它们的阶段；实验模式下使用空参数数组。
if [[ "$workflow_mode" == "synthetic" ]]; then
	train_workflow_args=(
		--synthetic-model "$synthetic_model"
		--ogden-mu "${ogden_mu[@]}"
		--ogden-alpha "${ogden_alpha[@]}"
		--mooney-rivlin-c10 "$mooney_rivlin_c10"
		--mooney-rivlin-c01 "$mooney_rivlin_c01"
		--synthetic-point-count "$synthetic_point_count"
		--synthetic-noise-std "$synthetic_noise_std"
		--synthetic-ut-range "$ut_stretch_min" "$ut_stretch_max"
		--synthetic-ps-range "$ps_stretch_min" "$ps_stretch_max"
		--synthetic-et-range "$et_stretch_min" "$et_stretch_max"
	)
	symbolic_workflow_args=(
		--synthetic-model "$synthetic_model"
		--ut-stretch-range "$ut_stretch_min" "$ut_stretch_max"
		--ps-stretch-range "$ps_stretch_min" "$ps_stretch_max"
		--et-stretch-range "$et_stretch_min" "$et_stretch_max"
	)
	fem_workflow_args=(--synthetic-model "$synthetic_model")
else
	train_workflow_args=()
	symbolic_workflow_args=()
	fem_workflow_args=()
fi

echo "Unified NN constitutive workflow"
echo "Workflow mode: $workflow_mode"
echo "Synthetic model: $synthetic_model"
echo "Symbolic feature space: $symbolic_feature_space"
echo "Valanis-Landel: $use_valanis_landel"
echo "Energy loss weight: $energy_loss_weight"
echo "Stress loss weight: $stress_loss_weight"

# set -e ensures that a failed stage stops the workflow immediately, so FEM
# will not run with stale symbolic output after an earlier failure.
# 任一阶段失败都会立即终止，避免后续阶段误用旧结果。
if [[ "$run_train" == true ]]; then
	"$script_dir/run_train.sh" "${common_args[@]}" "${train_hyperparameter_args[@]}" "${train_workflow_args[@]}"
fi

if [[ "$run_symbolic" == true ]]; then
	"$script_dir/run_symbolic.sh" "${common_args[@]}" "${symbolic_config_args[@]}" "${symbolic_workflow_args[@]}"
fi

if [[ "$run_fem" == true ]]; then
	"$script_dir/run_fem.sh" "${common_args[@]}" "${fem_workflow_args[@]}"
fi

echo "Unified workflow complete."
