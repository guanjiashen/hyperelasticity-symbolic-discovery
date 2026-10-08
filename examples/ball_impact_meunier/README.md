# 使用 Meunier 应变能的球冲击重算

使用正文图 4 的同一材料包，从 `../figure10_cann_csr/nn_cache/material.flat` 复制得到。
复用粗、细网格，以单独考察本构模型变化的影响。
两个网格均于 2026-10-07 使用 `/home/guanjs/OpenRadioss-latest-20251211`、其 GNU 双精度 Starter/Engine 及
`tools/userlib/examples/law291_userlib/libraduser_law291.so` 求解。
两次 Starter 运行均报告零错误、零警告；两次 Engine 均在 15 ms 正常结束，分别经历 3,065 和 4,351 个计算循环。

在已安装用户材料库的 Linux 主机上运行：

```bash
bash run_ball_wsl.sh all
MESH_SIZE=2.0 PREFIX=BALL_FINE bash run_ball_wsl.sh all
python3 postprocess_ball.py
```

脚本复用已有网格，仅在网格缺失时需要 GMSH。
请按本机安装位置设置 `OPENRADIOSS_ROOT` 和 `GMSH`。
原 Treloar 球冲击指标和图像不能作为 Meunier 结果使用。

后处理写入 `figs/meunier_ball_impact/`，从而保留旧图像。

`ball_impact_metrics.json` 保存响应指标和网格差异。
`ball_impact_qa.json` 保存能量检查，以及已保存动画帧的不变量和雅可比行列式范围，由 `audit_ball.py` 生成。
材料 SHA256 为 `65c940d93d2885595ca9e5a4c2fd23525b1f494cca3e45f2c5d75517c63d83c7`；Starter 报告的模型哈希为 `5c4133ef4ad76fa7`。
