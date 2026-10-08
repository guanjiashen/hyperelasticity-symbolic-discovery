# 超弹性应变能的符号发现

本仓库提供借助神经网络代理模型发现简洁超弹性应变能函数，并将其部署到 OpenRadioss 的代码和数据。

本次初始发布保留了选定的模型发现源码、数值数据和固定版本的结果。部分工作流仍使用历史路径，尚未完成完全可移植的端到端复现验证。

## 仓库内容

- `src/nn_invariant`：神经网络训练、符号回归和剪枝源码。
- `integrations/openradioss`：材料导出器与动态 USER01 材料库。
- `models/meunier`：训练好的网络和已部署的符号材料包。
- `examples`：带孔板与 Meunier 球冲击算例的输入文件、网格和结果。
- `benchmarks/material_point`：支撑 SI 表格的 Intel 单线程材料点性能审计。
- `results/figure_data`、`figures/scripts`：绘图输入数据及绘图、结果收集脚本。
- `MANIFEST.json`：整理发布时的源文件大小和 SHA256 校验值。

## 快速开始（Linux / WSL）

创建 Python 环境并安装声明的依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/nn_invariant/main.py --help
python integrations/openradioss/export_openradioss.py export-openradioss \
  --config models/meunier/SR_output/material_package.json \
  --output /tmp/meunier_material.flat --nu 0.495
```

使用 OpenRadioss 时，将 `OPENRADIOSS_ROOT` 设置为另行安装的源码或构建目录，再运行
`bash integrations/openradioss/userlib/build.sh` 构建用户材料库。
若在原求解器目录之外使用历史启动脚本，应将算例启动器的 `USERLIB` 设置为生成的库。

可通过以下命令重新运行标量及向量化的单线程 CPU 基准：

```bash
python benchmarks/material_point/audit_material_point.py
```

脚本将 `material_point_rerun.json` 写入原始审计结果所在目录。
耗时比取决于硬件及 Python 数值计算后端。PySR 会初始化 Julia 环境；随机搜索不保证重新发现的表达式与保存的模型完全相同。

OpenRadioss 需从 https://github.com/OpenRadioss/OpenRadioss 单独获取。
其许可证保留在 `LICENSES/` 中，该许可证不自动适用于所有作者自有文件；这些文件的许可仍待确定。

## 后续发布工作

1. 替换作者机器上的绝对路径，协调导出器内附实现与 `src/nn_invariant` 的版本。
2. 固定依赖和求解器版本，补充可移植的运行与构建说明。
3. 选择源码许可证，说明数据再分发条件。
4. 在干净环境中验证模型发现、材料包导出和求解器基本运行。
5. 发布经过验证的版本标签，并存档以获得持久标识符。

上述检查完成前，不应将此快照描述为完整的复现发布版。本快照未包含实验原始工作簿和论文扫描件。
