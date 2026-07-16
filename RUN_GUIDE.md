# HC-PINN Turing 补充实验运行指南

## 运行顺序（严格遵守依赖关系）

### 第一批：先跑这些（无依赖）
```bash
cd C:\Users\ROG\Desktop\HC_PINN_Turing

# 1. FDM 网格收敛性 (CPU, ~20s)
python experiments/fdm_convergence.py

# 2. 采样点敏感性 (GPU, ~6h)
python experiments/sensitivity_Nint.py

# 3. MLP 基线 (GPU, ~20min)
python experiments/mlp_baseline.py

# 4. Gamma 扫频 (GPU, ~5h)
python experiments/gamma_sweep.py

# 5. 频谱演化诊断 (GPU, ~30min)
python experiments/spectral_evolution.py
```
以上 2-5 可以开多个终端并行跑，不互依赖。

### 第二批：依赖第一批产出的模型权重
```bash
# 6. 多 seed 统计 (GPU, ~17h) —— 跑完这个才能填 Table 1
python experiments/multi_seed.py
```
这个最耗时，建议单独 overnight。

### 第三批：依赖 multi_seed 产出的 .pt 模型
```bash
# 7. NTK 分析
python analysis/ntk_analysis.py

# 8. 频谱残差分解
python analysis/spectral_residual.py
```

## 脚本-问题对应表

| 脚本 | 问题编号 | 预计耗时 | 产出 |
|:---|:---|:---|:---|
| `experiments/fdm_convergence.py` | #24 | 20s CPU | FDM 收敛性数据 |
| `experiments/sensitivity_Nint.py` | #9 | ~6h GPU | N_int 敏感性 |
| `experiments/mlp_baseline.py` | #22 | ~20m GPU | 数据驱动 MLP 基线 |
| `experiments/gamma_sweep.py` | #23 | ~5h GPU | γ 扫频 |
| `experiments/spectral_evolution.py` | #5 | ~30m GPU | 频谱演化热图 |
| `experiments/multi_seed.py` | #4 | ~17h GPU | 5-seed 统计 + 模型权重 |
| `analysis/ntk_analysis.py` | #6 | ~5m CPU | NTK 条件数 |
| `analysis/spectral_residual.py` | #13 | ~1m CPU | PDE 残差频谱分解 |
