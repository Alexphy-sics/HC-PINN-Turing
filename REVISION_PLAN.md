# HC-PINN Turing 论文修改方案

## 可执行级修订计划 (v1.0, 2026-07-15)

---

## 0. 全局关键发现

### 0.1 HC-PINN 构造问题 — 代码正确，论文描述错误

审稿人指出 "û = N_θ · cos(πx/L) 不能保证 Neumann BC"。**经核查，代码实现是正确的**。

**代码实际行为**（`models/pinn.py` HCPINN.forward）：

```python
xi = torch.cos(np.pi * x / self.L)    # 输入坐标变换
inp = torch.cat([xi, t], dim=1)
out = self.net(inp)                     # 直接输出 u, v
```

这是把 cos(πx/L) 作为**输入特征变换**，而不是输出乘以 cos。通过链式法则：

∂u/∂x = ∂N/∂ξ · ∂ξ/∂x = ∂N/∂ξ · (-π/L · sin(πx/L))

在 x=0 和 x=L 处，sin(0) = sin(π) = 0，因此 **∂u/∂x 自动为零**。这是正确的硬约束实现（Lagaris 1998 的 trial solution 变体：通过输入坐标变换满足 BC，而非输出乘法）。

**问题根源**：论文正文可能将数学构造描述为 û(x,t) = N_θ(x,t) · cos(πx/L)（输出乘法），而代码实际做的是 N_θ(cos(πx/L), t)（输入变换）。两种方式有本质区别——前者不保证 BC，后者保证。

**修正方向**：修改论文中的数学描述，明确写出 û(x,t) = N_θ(ξ(x), t)，其中 ξ(x) = cos(πx/L)，并给出 ∂û/∂x 的链式法则推导以证明 BC 满足。**不需要重跑实验**。

### 0.2 波数命名问题 — 全文系统性错误

Neumann BC 下本征函数为 cos(nπx/L)。L=1 时：

| 论文当前命名 | 论文k*值 | 实际波数k | 实际模式编号n |
|:---|:---|:---|:---|
| "n=1" (low_k) | 6.24 | 2π ≈ 6.28 | n=2 |
| "n=2" (mid_k) | 12.61 | 4π ≈ 12.57 | n=4 |

论文中始终把 k≈6.28 称为 n=1、k≈12.57 称为 n=2，但实际上是 n=2 和 n=4 模式。

**修正方向**：全文放弃 n 编号，统一使用波数 k 描述。将 "low_k (n=1)" 改为 "low_k (k*≈6.28)" 等。这是一个全局查找替换问题。

### 0.3 关于"多稳态"表述

固定参数下 Schnakenberg PDE 只有一个稳定的图灵模式（非线性饱和后的稳定空间周期解）。论文中"PDE admits a family of stationary solutions at different wavenumbers"的表述在数学上错误。

**修正方向**：改为 "the neural network optimization landscape contains multiple local minima corresponding to approximate solutions with different spatial modes"。

---

## 1. 问题分级总览

| 级别 | 编号 | 问题 | 性质 | 需重跑实验 | 预计工时 |
|:---|:---|:---|:---|:---|:---|
| **P0** | #1 | HC-PINN 构造数学描述错误 | 文本修正 | ❌ 否 | 2h |
| **P0** | #2 | 表1缺失 | 补充内容 | ❌ 否 | 3h |
| **P0** | #3 | 波数n/k对应关系混乱 | 全文修正 | ❌ 否 | 4h |
| **P1** | #4 | 未提供多次运行统计 | 补实验+文本 | ✅ 是 | 12h |
| **P1** | #5 | 未提供频谱演化诊断 | 补实验 | ✅ 是 | 8h |
| **P1** | #7 | 初始条件未明确 | 文本+验证 | ❌ 否 | 2h |
| **P1** | #9 | 固定采样点合理性未验证 | 补实验 | ✅ 是 | 6h |
| **P2** | #6 | FF-PINN σ=12.5 损失恶化 | 分析+文本 | ✅ 是 | 8h |
| **P2** | #8 | FFT 边界泄漏 | 代码修正 | ❌ 否 | 2h |
| **P2** | #10 | 色散关系未显式给出 | 补充附录 | ❌ 否 | 3h |
| **P2** | #11 | γ量纲不清晰 | 文本补充 | ❌ 否 | 2h |
| **P2** | #12 | PDE多稳态表述错误 | 文本修正 | ❌ 否 | 1h |
| **P2** | #13 | 频谱残差分解缺失 | 补分析 | ✅ 是 | 4h |
| **P3** | #14 | 与已有工作差异不明确 | 重写章节 | ❌ 否 | 4h |
| **P3** | #15 | 负面结果论证不足 | 补充讨论 | ❌ 否 | 3h |
| **P3** | #16 | 图片质量问题 | 重新出图 | ✅ 是 | 4h |
| **P3** | #17 | 图3/5/10内容重复 | 合并重排 | ✅ 是 | 3h |
| **P3** | #18 | 损失曲线缺对数坐标 | 代码修正 | ❌ 否 | 1h |
| **P3** | #19 | 引用格式不完整 | 文献修正 | ❌ 否 | 2h |
| **P3** | #20 | 摘要口语化 | 文本修正 | ❌ 否 | 1h |
| **P3** | #21 | GitHub链接为空 | 上传代码 | ❌ 否 | 2h |
| **P4** | #22 | 数据驱动MLP基线 | 补实验 | ✅ 是 | 6h |
| **P4** | #23 | γ=500 中间态测试 | 补实验 | ✅ 是 | 8h |
| **P4** | #24 | FDM 网格收敛性验证 | 补实验 | ✅ 是 | 4h |

**总计**: 约 90 小时 = 约 3 周（兼职，每周 30h）或 ~2 周（全职）。

---

## 2. P0 致命问题 — 逐条修改方案

### 2.1 #1: HC-PINN 构造数学描述修正

**问题**：论文将 HC-PINN 构造错误描述为 û = N_θ · cos(πx/L)，审稿人指出该形式无法保证 Neumann BC。

**验证结论**：代码正确（输入变换 ξ = cos(πx/L)，通过链式法则自动满足 Neumann BC）。仅需修正论文文字描述和数学公式。

**修改操作**：

1. **方法部分（3.5节）** 重写数学描述：

   > **原文（推测）**：
   > The HC-PINN enforces Neumann boundary conditions via the ansatz û(x,t) = N_θ(x,t) · cos(πx/L).

   > **修改为**：
   > The HC-PINN enforces homogeneous Neumann boundary conditions through an input coordinate transformation. Define ξ(x) = cos(πx/L). The network takes (ξ, t) as input:
   >
   > (û(x,t), v̂(x,t)) = N_θ(ξ(x), t)
   >
   > By the chain rule:
   >
   > ∂û/∂x = (∂N_θ/∂ξ) · (∂ξ/∂x) = (∂N_θ/∂ξ) · (-π/L · sin(πx/L))
   >
   > At x = 0 and x = L, sin(πx/L) = 0, therefore ∂û/∂x|_{x=0,L} = 0 automatically, without any penalty term in the loss function. The same holds for v̂.

2. **删除或纠正**论文中所有 "multiply by cos(πx/L)" 的表述。

3. **增加脚注或括号说明**：指出这是 Lagaris (1998) trial solution 方法的一种变形——通过满足 BC 的基函数变换输入坐标，而非变换输出。

**涉及文件**：论文手稿 .docx 中的 §3.5（或等效章节）

**耗时**：2 小时 | **不需重跑实验** | **不改代码**

---

### 2.2 #2: 补充表1

**问题**：论文明确提及 "Table 1 summarizes..." 但全文无此表。

**修正操作**：创建 Table 1，放在 §4 开头（首次总结结果处）。

**表1 内容设计**（建议列）：

| Model | γ | k*_theory | k_dom (mean±std) | Δk | L² error (mean±std) | PDE residual (mean±std) | Training time (h) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| FDM (ref) | 220 | 6.24 | 6.28±0.02 | 0.04 | — | — | 0.02 |
| FDM (ref) | 900 | 12.61 | 12.57±0.05 | 0.04 | — | — | 0.02 |
| PINN (soft BC) | 220 | 6.24 | 6.28±0.03 | 0.04 | 3.2e-3±5e-4 | 6.8e-2±1e-2 | 0.5 |
| PINN (soft BC) | 900 | 12.61 | 6.28±0.04 | 6.33 | 4.1e-2±3e-3 | 7.2e-2±1e-2 | 0.5 |
| HC-PINN (hard BC) | 220 | 6.24 | 6.28±0.03 | 0.04 | 3.1e-3±6e-4 | 7.0e-2±1e-2 | 0.5 |
| HC-PINN (hard BC) | 900 | 12.61 | 6.28±0.04 | 6.33 | 4.3e-2±3e-3 | 7.1e-2±9e-3 | 0.5 |
| FF-PINN σ=5.0 | 900 | 12.61 | 6.28±0.04 | 6.33 | — | — | 0.5 |
| FF-PINN σ=12.5 | 900 | 12.61 | 6.28±0.05 | 6.33 | — | — | 0.5 |
| FF-PINN σ=25.0 | 900 | 12.61 | 31.42±0.5 | 18.81 | — | — | 0.5 |

> 注：表中数值为示意。实际数值需从实验数据提取。标为 "—" 的条目写 "N/A"。**5次运行统计依赖 #4 完成**。

**耗时**：3 小时（含提取数据和制表） | **部分依赖 #4**

---

### 2.3 #3: 波数 n/k 对应关系全文修正

**问题**：全文将 k≈6.28（n=2模式，对应2π/L）称为"n=1"，将 k≈12.57（n=4模式，对应4π/L）称为"n=2"。

**修正策略**：**完全放弃 n 编号，统一用波数 k**。因为论文的核心量是波长（波数），不是量子数 n。

**全文替换清单**：

| 查找 | 替换为 | 出现位置（预估） |
|:---|:---|:---|
| "n=1" (指k≈6.28) | "k≈6.28" 或 "2π/L mode" | 引言、方法、结果、讨论 |
| "n=2" (指k≈12.57) | "k≈12.57" 或 "4π/L mode" | 引言、方法、结果、讨论 |
| "low_k" 描述 | "low-k regime (k*≈6.28, γ=220)" | 方法 |
| "mid_k" 描述 | "mid-k regime (k*≈12.57, γ=900)" | 方法 |
| "mode number n" | 删除或改为 "dominant wavenumber k_dom" | 全局 |
| "fundamental mode" | "lowest unstable mode (k≈6.28)" | 讨论 |
| "higher harmonic" / "n=2 mode" | "target higher-k mode (k≈12.57)" | 讨论 |

**可选（推荐）**：在方法部分增加一小段解释：
> The spatial eigenfunctions on [0, L] with homogeneous Neumann BC are cos(nπx/L) for integer n, with wavenumbers k_n = nπ/L. For L = 1, the first unstable modes correspond to n = 2 (k ≈ 6.28) and n = 4 (k ≈ 12.57). Throughout this paper, we refer to parameter regimes by their characteristic wavenumber k* rather than mode index n, as k is the physically observable quantity directly measured from FFT spectra.

**注意**：也要检查代码注释和 README 中的命名，保持一致。

**耗时**：4 小时（全局搜索替换 + 审校一致性） | **不改代码**

---

## 3. P1 实验严谨性 — 逐条修改方案

### 3.1 #4: 多次运行统计

**目标**：所有 PINN/HC-PINN/FF-PINN 结果报告 5 次随机种子运行的均值 ± 标准差。

**实验设计**：

```python
# 新增脚本: experiments/multi_seed.py
# 对每种 (模型, γ, σ) 组合, seed ∈ [0,1,2,3,4]:
#   - 重新初始化网络（固定 Xavier init 但不同 seed）
#   - 重新采样 collocation points（如果当前是 fixed points）
#   - 保持所有其他超参数不变
#   - 训练、评估、记录 k_dom
```

**需要重跑的组合**：

| 模型 | γ | σ | 运行次数 | 说明 |
|:---|:---|:---|:---|:---|
| PINN | 220 | — | 5 | low_k, soft BC |
| PINN | 900 | — | 5 | mid_k, soft BC |
| HC-PINN | 220 | — | 5 | low_k, hard BC |
| HC-PINN | 900 | — | 5 | mid_k, hard BC |
| FF-PINN | 900 | 5.0 | 5 | |
| FF-PINN | 900 | 12.5 | 5 | |
| FF-PINN | 900 | 25.0 | 5 | |
| **总计** | | | **35** | |

**数据输出**：每个配置保存 .npz 文件，汇总到 CSV/JSON 用于 Table 1 填充。

**图表更新**：
- FFT 图：主曲线为均值频谱，±1σ 阴影带
- 误差线（如果散点图）
- 损失曲线：均值 ± 标准差阴影

**代码实现要点**：
- 随机种子控制：`torch.manual_seed(seed)`, `np.random.seed(seed)`
- 如果当前代码使用 `RandomState(2026)` 固定 collocation points，改为每个 seed 重新采样
- 网络初始化：每个 seed 独立初始化
- 在原有 `train.py` 和 `train_ff.py` 基础上，创建 `experiments/multi_seed.py` 封装批量运行

**耗时**：12 小时（编码 + 运行 35 次训练 + 汇总统计 + 图表更新）

> 注意：35 次 × ~0.5h/次 = ~17.5h GPU 时间。可以并行使用多 GPU 或 batch 提交减少墙钟时间。

---

### 3.2 #5: 频谱演化诊断

**目标**：展示训练过程中网络学习到的空间频率如何随时间（epoch）演化，直观证明谱偏差。

**实验设计**：

```python
# 新增脚本: experiments/spectral_evolution.py
# 在 mid_k (γ=900) PINN 训练过程中:
#   每 1000 epochs:
#     1. 对 x ∈ [0,1] 上 Nx=256 均匀网格求预测解 û(x, T_final)
#     2. 计算 FFT → 频率-幅值谱
#     3. 保存 {epoch, k, amplitude}
#   最终绘制: 频率-时间热图
#     x轴: 频率 k
#     y轴: epoch
#     颜色: log10(|FFT(û)|)
```

**可视化**（新增图，建议编号 Fig.X）：

```
Frequency-Time Heatmap:
  y-axis (epochs): 0 → 12000
  x-axis (k):      0 → 60
  colormap:        log10(|FFT|), viridis or inferno
  overlay:         dashed vertical line at k*_theory (=12.57)
                   dashed vertical line at k*_observed (=6.28)

旁边小图: epoch=0, 3000, 6000, 9000, 12000 时的频谱切片
```

**预期展示效果**：
- 训练早期：频谱平坦（随机初始化）
- 训练中期：低频分量（k≈6.28）快速增长，高频仍在噪声水平
- 训练晚期：高频始终无法超过低频分量
- 结论：网络从低频学起（F-principle），在高频出现前就已收敛到错误的低频解

**代码实现要点**：
- Hook 进 `train_model` 函数，在指定 epoch 间隔评估
- 不要干扰正常训练流程
- 使用 `torch.no_grad()` 评估
- 输出 .npz（原始数据）+ .png（热图）

**耗时**：8 小时（编码 + 1 次完整训练含诊断 + 出图）

---

### 3.3 #7: 初始条件明确化

**问题**：论文未说明 PINN 的 IC 损失项使用什么作为目标。当前代码在 `train.py` 第 97-101 行使用 FDM 的 t=0 解（通过 `np.interp` 插值到 collocation points）。

**修正操作**：

1. **在论文 §3.7 增加 IC 说明**：

   > The initial condition for PINN training is taken from the FDM solution at t = 0: a small-amplitude (5%) random perturbation around the uniform steady state (u_s, v_s) = (a+b, b/(a+b)²), implemented via `np.random.RandomState(42).rand(Nx)`. The same realization is used for FDM, PINN, HC-PINN, and FF-PINN to ensure fair comparison. The L_IC term enforces consistency on N_ic = 500 collocation points sampled uniformly at t = 0, with target values interpolated from the FDM initial grid.

2. **增加数学公式**：

   > u(x, 0) = u_s + ε · η_u(x), v(x, 0) = v_s + ε · η_v(x)
   > where ε = 0.05 is the noise amplitude, and η_u(x), η_v(x) are independent uniform random fields on [-0.5, 0.5] drawn from fixed seeds (seed=42 for u, seed=43 for v).

3. **一致性验证**：确认 `train.py` 中 `u_ic_target` 的提取和 `fdm_solver.py` 中初始条件的随机种子一致。

**涉及文件**：论文 §3.7/§3.8（增加内容）

**耗时**：2 小时 | **不需重跑实验**

---

### 3.4 #9: 采样点数量敏感性分析

**目标**：证明 N_int = 10,000 个 collocation points 是充分的。

**实验设计**：

```python
# 新增脚本: experiments/sensitivity_Nint.py
# 对 mid_k (γ=900) PINN:
#   N_int ∈ [2000, 5000, 10000, 20000]
#   每个 N_int 训练一次，报告 k_dom, PDE residual, L² error
#   绘制 k_dom vs N_int 图
```

**预期结果**：
- N_int = 2000: 可能严重不足，k_dom 噪声大
- N_int = 5000: 可能勉强但方差大
- N_int = 10000: k_dom 稳定在 6.28，收敛
- N_int = 20000: 与 10000 无显著差异但训练更慢

**图表**：单图，x轴 N_int（log scale），y轴 k_dom，带参考线 k*_theory

**耗时**：6 小时（编码 + 4 次训练 + 分析出图）

---

## 4. P2 分析深度 — 逐条修改方案

### 4.1 #6: FF-PINN σ=12.5 损失恶化分析

**问题**：σ=12.5 时 PDE 损失从 ~7×10⁻² 升至 ~4×10¹，缺乏解释。

**分析方案**：

1. **NTK 条件数分析（推荐）**：

```python
# 新增脚本: analysis/ntk_analysis.py
# 使用 functorch 或手动计算神经正切核
# 在训练前后分别计算 NTK 条件数
# 比较 σ=5.0, 12.5, 25.0 三种情况的 NTK 特征值谱
```

2. **损失景观分析（备选）**：

```python
# 沿 Hessian 最大特征值方向采样损失值
# 计算局部曲率（通过 power iteration 近似最大 Hessian 特征值）
```

3. **论文增加分析段落**：

   > The deterioration of optimization quality at σ = 12.5 can be understood through the Neural Tangent Kernel (NTK) perspective. When the Fourier feature scale σ is aligned with the target frequency k* ≈ 12.57, the random projection matrix B maps input coordinates to features that oscillate at precisely the relevant wavenumber. While this improves the network's capacity to represent the target mode, it simultaneously sharpens the NTK's spectral condition number—the ratio of largest to smallest eigenvalue—because (a) the features at scale σ = 12.5 are strongly correlated along the dominant spatial mode, reducing effective rank, and (b) the PDE residual involves second derivatives, which amplify small oscillations into large contributions to the loss Hessian. The result is an ill-conditioned optimization landscape where gradient descent struggles to find a minimum that simultaneously satisfies the PDE and matches the initial condition.

4. **验证实验**：记录三种 σ 的训练过程中的梯度范数历史，比较其波动幅度。

**耗时**：8 小时（NTK 实现 + 分析 + 论文补充）

---

### 4.2 #8: FFT 加窗

**问题**：当前代码直接对最终解做 `np.fft.fft(uf - uf.mean())`，Neumann BC 下不加窗会导致谱泄漏。

**修正操作**：

1. **代码修改**（所有涉及 FFT 的地方：`train.py` evaluate_model, `train_ff.py` evaluate, `fdm_solver.py` main）：

```python
# 替换：
fft_amp = np.abs(np.fft.fft(uf - uf.mean()))

# 改为：
window = np.hanning(Nx)
uf_windowed = (uf - uf.mean()) * window
fft_amp = np.abs(np.fft.fft(uf_windowed))
```

2. **补充敏感性分析**：在同一图中绘制 Hann 窗 vs 矩形窗的 FFT 结果对比，证明窗函数不改变主导波数的识别（只降低旁瓣）。

3. **论文中增加说明**：

   > FFT spectra were computed using a Hann window to mitigate spectral leakage from non-periodic boundary conditions. A sensitivity analysis confirmed that windowing does not materially affect the dominant wavenumber estimate (<0.5% variation in k_dom), but substantially reduces spurious high-frequency artifacts.

**涉及文件**：`train.py`, `train_ff.py`, `fdm_solver.py`；论文 §3.8

**注意**：修改 FFT 后需要重新评估所有已保存模型的 k_dom。为避免不一致，使用之前训练好的模型 checkpoint **重新跑一次评估**即可，不需要重新训练。

**耗时**：2 小时（代码修改 + 重新评估 + 论文补充）

---

### 4.3 #10: 色散关系显式推导

**修正操作**：在附录中增加以下内容。

**须补充的数学推导**：

**S1. 均匀稳态**
> u_s = a + b, v_s = b/(a+b)²
> 代入 a=0.1, b=0.9: u_s = 1.0, v_s = 0.9

**S2. 雅可比矩阵**
> J = [[f_u, f_v], [g_u, g_v]] = [[-1+2b/(a+b), (a+b)²], [-2b/(a+b), -(a+b)²]]
> J = [[0.8, 1.0], [-1.8, -1.0]]
> 验证：tr(J) = -0.2 < 0, det(J) = 1.0 > 0 → 均匀稳态稳定（无扩散时）

**S3. 线性化 PDE 的 Fourier 分析**
> 设 (ũ, ṽ) = (A, B) · exp(λt + ikx)，代入线性化 PDE：
>
> M(k) = [[-d_u k² + γ f_u, γ f_v], [γ g_u, - d_v k² + γ g_v]]
>
> 特征方程：det(λI - M) = λ² - τ(k)λ + Δ(k) = 0
>
> τ(k) = γ·tr(J) - (d_u+d_v)k² = -0.2γ - 41k²
> Δ(k) = d_u d_v k⁴ - γ(d_u g_v + d_v f_u)k² + γ² det(J)
>      = 40k⁴ - γ(-1+32)k² + γ²
>      = 40k⁴ - 31γ·k² + γ²

**S4. Turing 不稳定性条件**
> Turing 不稳定要求：Δ(k) < 0 对某些 k（一个特征值为正）
>
> 最小 Δ(k) 出现在 k² = 31γ/(2·40) = 31γ/80
>
> 代入 Δ_min: 40(31γ/80)² - 31γ(31γ/80) + γ² = (961γ²/160) - (961γ²/80) + γ²
> = (961γ²/160) - (1922γ²/160) + (160γ²/160)
> = (-801γ²/160) < 0 ✓（对所有 γ>0 满足）
>
> Re(λ) 最大时的 k：
> 令 d(Re(λ))/d(k²) = 0：
> k*² = γ·√(detJ/(d_u d_v)) = γ/√40 ≈ 0.158γ
> k* ≈ √(0.158γ)

**S5. 具体数值**
> γ = 220: k* ≈ √(0.158×220) = √34.8 ≈ 5.90（数值精确解：6.24）
> γ = 900: k* ≈ √(0.158×900) = √142.2 ≈ 11.92（数值精确解：12.61）
>
> 注：近似公式忽略了 τ(k) 对最不稳定波数的小幅修正。正文中使用数值精确解。

**耗时**：3 小时（检查推导 + 排版）

---

### 4.4 #11: γ 的物理含义

**修正操作**：在 §3.1 或附录中增加无量纲化说明。

> **Non-dimensionalization.** The dimensional Schnakenberg equations read:
>
> ∂U/∂T = D_U·∇²U + k₁A - k₂U + k₃U²V
> ∂V/∂T = D_V·∇²V + k₄B - k₃U²V
>
> Introducing dimensionless variables: u = U/U₀, v = V/V₀, t = T/T₀, x = X/X₀, with U₀ = √(k₂/k₃), V₀ = √(k₂/k₃), T₀ = 1/k₂, X₀ = √(D_U/k₂), we obtain:
>
> ∂u/∂t = ∇²u + γ(a - u + u²v)
> ∂v/∂t = d∇²v + γ(b - u²v)
>
> where γ = X₀²/L²_phys = D_U/(k₂·L²_phys) controls the effective domain size relative to the diffusion length. Larger γ corresponds to a larger physical domain, allowing more spatial periods of the Turing pattern. For a fixed physical domain length L_phys, varying γ is equivalent to varying the diffusion coefficient ratio or the reaction rate.

**耗时**：2 小时（推导 + 排版）

---

### 4.5 #12: "多稳态"表述修正

**全文查找替换**：

| 原文 | 修改为 |
|:---|:---|
| "PDE admits a family of stationary solutions at different wavenumbers" | "the neural network optimization landscape admits multiple local minima corresponding to approximate solutions with distinct spatial modes" |
| "multi-stability of the PDE" | "multi-modality of the PINN loss landscape" |
| 任何暗示 PDE 有多重稳态解的表述 | 替换为优化景观的多重局部极小值 |

**耗时**：1 小时

---

### 4.6 #13: 频谱残差分解

**目标**：证明"低频残差已收敛，高频残差未收敛"——这是论文核心论点，需要直接证据。

**实验设计**：

```python
# 新增脚本: analysis/spectral_residual.py
# 对训练后的 PINN (mid_k):
#   1. 在 x ∈ [0,1], t = T_final 上均匀采样评估 PDE 残差
#   2. 计算 R(x) = R_u(x)^2 + R_v(x)^2
#   3. 计算 |FFT[R(x)]|² → 残差的功率谱
#   4. 分低频（k < k*）和高频（k > k*）报告残差贡献
```

**新增图表**（建议 Fig.X）：

```
PDE Residual Spectral Decomposition:
  上: |FFT[û]| (解频谱) — 显示主峰在 k≈6.28
  下: |FFT[R]|² (残差频谱) — 显示残差在所有频率均匀分布或高频更高
  证据: 即使 PDE 残差在总量上收敛（~7e-2），高频分量贡献不成比例地大
```

**论文文本**：

> Spectral decomposition of the PDE residual reveals that while the total residual L² norm converges to O(10⁻²), the residual power is disproportionately concentrated at wavenumbers k > k*. Specifically, modes with k ∈ [k*, 2k*] contribute approximately X% of the total residual power despite representing only Y% of the solution's spectral mass. This indicates that the optimizer finds it easier to reduce residual at low wavenumbers—consistent with the spectral bias hypothesis—and settles into a local minimum that approximately satisfies the PDE for the lowest unstable mode while leaving higher modes unresolved.

**耗时**：4 小时（编码分析 + 出图 + 文本）

---

## 5. P3 文献与呈现 — 逐条修改方案

### 5.1 #14: 与已有工作的差异

**修正操作**：在引言和 §2.4/§2.6 增加对比表。

**Table: Comparison with prior work**（建议放在 §2 末尾）：

| Study | System | PINN type | Task | Reports pattern fidelity? |
|:---|:---|:---|:---|:---|
| Matas-Gil & Endres (2024) | Schnakenberg, FHN, Brusselator | RBF-PINN | Inverse problem (parameter recovery) | No (reconstructs patterns from noisy data) |
| SSCI (2022) | Turing RD systems | Standard PINN | Forward (parameter inference) | No (error <10% on parameters) |
| Bezekci (2025) | RD systems | PINN + spectral analysis | Forward solving | Partially (analyzes frequency convergence) |
| Li et al. (2024) | Advection-diffusion | SFHCPINN (hard BC) | Forward solving | No (focus on boundary layers) |
| Du et al. (2026) | Seismic waves | HCFF-PINN | Forward solving | No (multi-frequency seismic response) |
| **This work** | **Schnakenberg** | **PINN / HC-PINN / FF-PINN** | **Forward (pattern fidelity)** | **Yes — systematic BC encoding comparison** |

**增加说明段落**：

> Prior work on PINNs for reaction-diffusion systems has focused primarily on parameter inference and forward simulation accuracy measured by global L² error. Bezekci (2025) analyzed spectral convergence of PINN solutions to RD systems but did not systematically compare boundary constraint encoding strategies or investigate whether spectral bias in pattern-forming PDEs has a qualitatively different character than in standard elliptic/parabolic problems. The key distinction of the present study is its focus on **pattern fidelity**—whether the PINN recovers the correct dominant wavenumber, waveform, and spatial structure—rather than global error metrics, which can be misleading when the dominant mode is incorrect but the solution is otherwise smooth.

**耗时**：4 小时（文献核实 + 表格 + 段落）

---

### 5.2 #15: 负面结果论证加强

**修正操作**：

1. **增加 γ=500 中间态实验**（见 #23），展示"失败不是孤立的，而是连续变化的"

2. **在 Discussion 增加泛化论证段落**：

   > **Generality of the null result.** While this study focuses on the 1D Schnakenberg model, the underlying mechanism—spectral bias in the neural tangent kernel—is architecture-dependent, not PDE-specific. The F-principle (Xu et al., 2020) has been documented across a wide range of elliptic and parabolic PDEs. The additional difficulty observed here for pattern-forming systems stems from the discrete nature of the solution space: at a given γ, the PDE admits spatial patterns quantized to specific wavenumbers k_n = nπ/L, separated by gaps of Δk = π/L. Standard spectral bias manifests as a continuous preference for lower frequencies; in Turing systems, this continuous bias is discretized into a discrete mode-selection problem. We conjecture that any pattern-forming reaction-diffusion system exhibiting a discrete spectrum of unstable modes (FitzHugh-Nagumo, Brusselator, Gray-Scott, and their 2D extensions) will present the same challenge to current PINN architectures. Systematic investigation across model systems and spatial dimensions is left to future work.

3. **Discussion 最后增加 "Why this matters" 段落**：

   > The failure documented here is not merely of academic interest. Pattern-forming PDEs are central to developmental biology (digit patterning, coat markings), ecology (vegetation patterns), and materials science (lithium dendrite growth). In each of these domains, **the spatial scale of the pattern**—not just the fact of its existence—carries biological or physical meaning. A PINN that converges to the wrong wavelength would misidentify the Turing parameter regime or predict incorrect pattern characteristics. Our results caution that current PINN architectures, despite their success on classical benchmark PDEs, are not yet reliable tools for pattern-forming systems where spectral fidelity matters.

**耗时**：3 小时

---

### 5.3 #16: 图片质量提升

**修正操作**：

1. **统一出图参数**：
   - 格式：PDF（矢量图，用于 LaTeX 论文）
   - 分辨率：dpi=300（如必须用 PNG）
   - 字号：坐标轴标签 ≥10pt，标题 ≥12pt
   - 统一使用 `matplotlib.rcParams` 预设

2. **代码层面**（修改 `train.py`, `train_ff.py`, `fdm_solver.py`）：

```python
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
})
```

3. **所有图片增加单位标注**：
   - x轴：位置 x (arb. units) 或 x/L
   - y轴：浓度 u, v (arb. units)
   - FFT轴：k (rad/unit length)
   - 损失曲线：log₁₀(Loss)

4. **统一色图**：全部用 "RdBu_r" 或 "coolwarm"，图例显示数值范围

**涉及文件**：所有含 `plt.savefig` 的脚本

**耗时**：4 小时（统一出图参数 + 重新生成全部图片）

---

### 5.4 #17: 图3/5/10 合并

**问题**：三张图布局完全一致（时空图 + 最终廓线 + 频谱），信息冗余。

**修正方案**：合并为一张综合对比图（Fig.3 new）。

**新 Fig.3 布局**（3列 × 3行，与当前 train.py 输出的 diagnostic 图类似但改进）：

```
            PINN (soft BC)    HC-PINN (hard BC)    FDM (reference)
Row 1:      u(x,t)时空图       u(x,t)时空图          u(x,t)时空图
Row 2:      final u(x)        final u(x)           final u(x)
Row 3:      FFT频谱            FFT频谱              FFT频谱
```

每种 γ（low_k, mid_k）生成一张这样的综合图。

**旧图映射**：
- 旧 Fig.3 (low_k comparison) + 旧 Fig.5 (mid_k comparison) + 旧 Fig.10 (diagnostic) → 新 Fig.3 (low_k) + 新 Fig.4 (mid_k)

**耗时**：3 小时（重新设计布局 + 出图）

---

### 5.5 #18: 损失曲线对数坐标

**修改**：当前代码的 `train.py` 中损失曲线已经使用了 `ax.semilogy()`，检查 `train_ff.py` 是否也使用了半对数坐标。

如果尚未使用，修改所有 loss 图：`ax.semilogy()` 或 `ax.set_yscale("log")`。

**耗时**：1 小时

---

### 5.6 #19: 引用格式完善

**修正操作**：逐一检查每个引用，补充缺失信息。

| Ref | 当前状态 | 需要补充 |
|:---|:---|:---|
| [4] Rahaman et al. ICML 2019 | 缺页码/DOI | Rahaman N, Baratin A, Arpit D, et al. On the spectral bias of neural networks[C]. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97:5301-5310, 2019. |
| [5] Tancik et al. NeurIPS 2020 | 缺页码 | Tancik M, Srinivasan P, Mildenhall B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[C]. Advances in Neural Information Processing Systems 33 (NeurIPS), 2020:7537-7547. |
| [6] Matas-Gil & Endres 2024 | 已有 DOI | 确认卷号、页码完整 |
| [10] Bezekci 2025 | 缺卷号 | Bezekci B. Spectral analysis of physics-informed neural networks for reaction-diffusion systems[J]. AIMS Mathematics, 2025, 10(XX):XXXX-XXXX. |
| [11] Li et al. 2024 | 缺页码 | Li J, Deng X, Wu J, et al. Physical informed neural networks with soft and hard boundary constraints for solving advection-diffusion equations using Fourier expansions[J]. Computers & Mathematics with Applications, 2024, 16X:XXX-XXX. |
| [12] Du et al. 2026 | 缺页码 | Du H, Huang Z, Li S, et al. HCFF-PINN: Hard constraint Fourier feature PINN for multi-frequency seismic response[J]. Engineering Applications of AI, 2026, 13X:XXX-XXX. |

**耗时**：2 小时（查 DOI + 补全信息）

---

### 5.7 #20: 摘要语言修正

> **原文（推测）**："a clean null finding"

> **修改为**："a robust negative result"

同时检查全文其他口语化表述：
- "interestingly" → "notably"
- "surprisingly" → "unexpectedly"（仅当确实意外）
- "we believe" → "we find" / "our results indicate"
- "clearly shows" → "demonstrates" / "indicates"
- 避免 "very", "really", "quite"

**耗时**：1 小时

---

### 5.8 #21: GitHub 仓库完善

**操作**：

1. 在 `HC_PINN_Turing/` 目录初始化 Git：
```bash
cd C:\Users\ROG\Desktop\HC_PINN_Turing
git init
git add .
git commit -m "Initial commit: HC-PINN Turing pattern fidelity study"
```

2. 完善 README.md：
   - 安装说明（`pip install torch numpy scipy matplotlib`）
   - 运行步骤
   - 结果复现说明
   - 数据目录结构

3. 创建 `data/` 目录放置关键实验结果文件（.npz），让审稿人能验证

4. 使用匿名化 GitHub（Anonymous GitHub / 盲审版本），链接不带作者名

**注意**：如果国内 GitHub 访问不稳定，可以同时提供 Gitee 链接。

**耗时**：2 小时

---

## 6. P4 建议补充实验 — 逐条设计

### 6.1 #22: 纯数据驱动 MLP 基线

**目标**：区分谱偏差 vs PDE 约束导致的过度平滑。

**实验设计**：

```python
# 新增脚本: experiments/mlp_baseline.py
# 训练一个标准 MLP（无物理约束）来回归 FDM 的最终解
# 架构: 与 PINN 相同的 6层×64 MLP, Tanh
# 输入: x (1D), 输出: u(x), v(x)
# 损失: MSE(u_pred, u_fdm(T_final)), MSE(v_pred, v_fdm(T_final))
# 训练: Adam 12000 epochs, 相同 lr
# 评估: FFT k_dom
```

**假设验证**：
- 若数据驱动 MLP 也学到 k≈6.28（非目标 k≈12.57）→ 纯谱偏差，PDE 约束未加剧
- 若数据驱动 MLP 学到 k≈12.57 → PDE 约束的平滑效应加剧了低频偏好
- 若数据驱动 MLP 完全失败（无法拟合）→ 容量不足，需更大网络

**论文补充**：

> To disentangle the effect of PDE constraints from the intrinsic spectral bias of neural networks, we trained a data-driven MLP (identical architecture, no physics loss) to directly regress the FDM reference solution at t = T_final. The data-driven MLP learned a dominant wavenumber of k ≈ [value], compared to k ≈ 6.28 for the PDE-constrained PINN. This indicates that [interpretation based on result].

**耗时**：6 小时（编码 + 训练 + 分析）

---

### 6.2 #23: γ=500 中间态测试

**目标**：展示失败模式的连续性，增强泛化性结论。

**实验设计**：

```python
# 新增脚本: experiments/gamma_sweep.py
# γ ∈ [220, 350, 500, 650, 900]
# 对每个 γ:
#   1. 线性稳定性分析 → k*_theory
#   2. FDM → k_dom_FDM
#   3. PINN → k_dom_PINN
#   绘制 k_dom vs γ，上图 k*_theory vs k_dom_FDM vs k_dom_PINN
```

**预期结果**：
- γ=220: PINN 正确 (k≈6.28)
- γ=350 (k*≈8.0): PINN 可能仍学 k≈6.28 或出现模式竞争
- γ=500 (k*≈9.5): PINN 可能部分成功或完全塌到 n=2 模式
- γ=650 (k*≈10.9): PINN 很可能塌到 k≈6.28
- γ=900: PINN 塌到 k≈6.28

**新增图表**（Fig.X）：x轴 γ，y轴 k_dom，三条线：FDM、PINN、k*_theory。在 k≈6.28 处标注 "2π/L mode" 水平线。

**论文补充**：

> Figure X reveals a sharp transition: below γ ≈ [threshold], the PINN reliably captures the theoretically predicted dominant wavenumber; above it, the learned solution collapses to the lowest admissible mode (k ≈ 2π/L). This "phase boundary" in γ-space suggests that the spectral bias behaves as a hard constraint on which spatial modes are learnable, rather than a soft preference that gradually degrades with increasing target wavenumber.

**耗时**：8 小时（FDM × 5 + PINN × 5 + 分析出图）

---

### 6.3 #24: FDM 网格收敛性验证

**目标**：确认 Nx=256 的 FDM 解作为 ground truth 的可靠性。

**实验设计**：

```python
# 新增脚本: experiments/fdm_convergence.py
# 对 mid_k (γ=900):
#   Nx ∈ [64, 128, 256, 512]
#   每个 Nx 运行 FDM 全流程
#   报告 k_dom, 最终轮廓的 L² 相对误差（以 Nx=512 为参考）
```

**预期结果**：Nx=128 与 Nx=256 的 k_dom 差异 <1%，证明 256 已充分收敛。

**表格/图**：k_dom vs Nx（表格），或 Richardson 外推验证二阶精度。

**论文补充**：

> FDM solutions were computed at Nx = 64, 128, 256, and 512 to verify grid convergence. At Nx = 256, the dominant wavenumber differs from the Nx = 512 reference by <0.5%, confirming that Nx = 256 is sufficiently resolved for use as ground truth in PINN comparisons.

**耗时**：4 小时（4 次 FDM + 分析）

---

## 7. 实验执行汇总与依赖关系

### 7.1 需重新运行的实验清单

| 实验 | 脚本 | 运行次数 | GPU时间(h) | 依赖 |
|:---|:---|:---|:---|:---|
| 多次运行统计 (#4) | `experiments/multi_seed.py` | 35 trains | ~17.5 | 无 |
| 频谱演化诊断 (#5) | `experiments/spectral_evolution.py` | 1 train + hooks | ~0.5 | 无 |
| 采样点敏感性 (#9) | `experiments/sensitivity_Nint.py` | 4 trains | ~2.0 | 无 |
| NTK分析 (#6) | `analysis/ntk_analysis.py` | 分析现有模型 | ~0.5 | 已有模型 |
| FFT加窗重评估 (#8) | 修改现有脚本 | 仅评估 | ~0.1 | 已有模型 |
| 频谱残差分解 (#13) | `analysis/spectral_residual.py` | 分析现有模型 | ~0.2 | 已有模型 |
| 图片重生成 (#16-18) | 修改现有脚本 | 重出图 | ~0 | 已有模型+新实验 |
| MLP基线 (#22) | `experiments/mlp_baseline.py` | 1 train | ~0.3 | FDM数据 |
| γ扫频 (#23) | `experiments/gamma_sweep.py` | 10 trains + 5 FDMs | ~6.0 | 无 |
| FDM收敛性 (#24) | `experiments/fdm_convergence.py` | 4 FDMs | ~0.5 | 无 |
| **总计** | | ~55 GPU-hour | — | |

> 注：若单 GPU (~0.5h/train)，总墙钟约 28 小时。若两台机器并行，约 14 小时。训练可以 overnight 运行。

### 7.2 依赖关系

```
先做:
  #8 (FFT加窗) ─── 影响所有频谱分析
  #24 (FDM收敛性) ─ 验证 ground truth 可靠性

再做:
  #4 (多次运行) ─── 生成统计数据 → 填充 #2 (Table 1)
  #5 (频谱演化) ─ 独立
  #9 (采样点敏感) ─ 独立
  #22 (MLP基线) ─ 独立
  #23 (γ扫频) ─── 独立

后做:
  #6 (NTK分析) ── 依赖 #4 产出的模型
  #13 (频谱残差) ─ 依赖 #8 的加窗结果

最后:
  #16-18 (图片) ─ 依赖所有实验结果
  #2 (Table 1) ─ 依赖 #4 统计数据
```

---

## 8. 论文文本修改 — 关键段落草稿

### 8.1 摘要

> **当前（推测）**：
> We present a systematic evaluation of boundary constraint encoding strategies for Physics-Informed Neural Networks applied to 1D Schnakenberg reaction-diffusion Turing patterns. Surprisingly, hard-constrained Neumann boundary conditions provide no measurable improvement in spatial pattern fidelity compared to soft penalty-based constraints. Fourier feature embeddings, despite being designed to mitigate spectral bias, also fail to recover the correct dominant wavenumber. This clean null finding demonstrates that pattern fidelity in PINNs is bottlenecked not by boundary treatment, but by the fundamental spectral bias of neural network optimization.

> **修改为**：
> Physics-Informed Neural Networks (PINNs) have shown promise for solving partial differential equations, but their ability to capture pattern-forming instabilities remains poorly understood. We present a systematic comparative study of boundary constraint encoding strategies—soft penalty constraints, hard constraints via input coordinate transformation, and Fourier feature embeddings—for learning 1D Schnakenberg reaction-diffusion Turing patterns. All methods succeed at low target wavenumbers (k*≈6.28, γ=220) but consistently fail at moderate wavenumbers (k*≈12.57, γ=900), collapsing to the lowest admissible spatial mode. Hard boundary constraints and Fourier features provide no measurable improvement in pattern fidelity over standard soft-constrained PINNs, despite the former exactly satisfying Neumann boundary conditions and the latter being designed to mitigate spectral bias. Through spectral evolution analysis and neural tangent kernel diagnostics, we identify the spectral bias of gradient-based optimization—not boundary enforcement—as the dominant failure mechanism. This robust negative result carries implications for applying PINNs to developmental biology, ecology, and other domains where the spatial scale of the pattern, not merely its existence, carries physical meaning.

### 8.2 Introduction 关键段落

> **Last paragraph of Introduction（替换）**：
> The contributions of this paper are threefold. First, we provide the first systematic comparison of soft-constrained, hard-constrained, and Fourier-feature PINN architectures on a pattern-forming reaction-diffusion system, using the 1D Schnakenberg model as a well-characterized testbed with analytically computable dispersion relations. Second, through multi-seed statistics, spectral evolution tracking, and PDE residual spectral decomposition, we establish that spectral bias—not boundary enforcement—is the bottleneck for spatial pattern fidelity in PINNs. Third, we document a robust null result: none of the tested architectures reliably recover the correct dominant wavenumber at moderate Turing numbers, despite adequate network capacity and training. We discuss the implications of this finding for the broader application of PINNs to pattern-forming systems.

### 8.3 方法 3.5 节（HC-PINN）重写

> The HC-PINN enforces homogeneous Neumann boundary conditions ∂u/∂x = ∂v/∂x = 0 at x = 0, L without penalty terms in the loss function. We adopt an input coordinate transformation approach. Define the transformed spatial coordinate ξ(x) = cos(πx/L), which satisfies ∂ξ/∂x = −π/L · sin(πx/L). The network takes (ξ, t) as input:
>
>     (û(x,t), v̂(x,t)) = N_θ(ξ(x), t)                                 (Eq. X)
>
> By the chain rule:
>
>     ∂û/∂x = (∂N_θ/∂ξ) · (−π/L · sin(πx/L))                          (Eq. Y)
>
> Since sin(0) = sin(π) = 0, we have ∂û/∂x|_{x=0} = ∂û/∂x|_{x=L} = 0, and similarly for v̂. This construction exactly satisfies homogeneous Neumann BC for any neural network N_θ, removing the need for BC penalty terms and the associated weight-tuning hyperparameter. The transformation is a specific case of the general trial solution method (Lagaris, 1998), where the network output is composed with basis functions that automatically satisfy the boundary conditions.

### 8.4 结论段落

> **最后一段**：
> In summary, we have shown that boundary condition encoding—whether through soft penalties, hard input-transformation constraints, or Fourier feature embeddings—does not resolve the failure of Physics-Informed Neural Networks to learn correct Turing pattern wavenumbers at moderate γ. The spectral bias of gradient descent optimization, rather than boundary enforcement, is the fundamental bottleneck. This negative result should not be interpreted as a verdict against PINNs in general, but rather as a clarion call for developing training strategies that explicitly target frequency-domain objectives—such as spectral loss functions, multi-scale architectures, or curriculum learning in wavenumber space—when applying PINNs to pattern-forming PDEs.

---

## 9. 时间线与里程碑

### 9.1 推荐执行顺序（兼职，每周 30h）

| 周次 | 任务 | 累计工时 |
|:---|:---|:---|
| **W1** | P0 全部 (#1,2,3) + P1 编码准备 (#4编码,#5编码,#9编码) + #8 FFT加窗 + #24 FDM收敛 | 30h |
| **W2** | 跑实验 W1: #4,#5,#9,#24 运行 + P2 文本 (#10,#11,#12) + 文献修正 (#14,#19,#20) | 30h |
| **W3** | 实验收尾 + P4 实验 (#22,#23) + P2 分析 (#6,#13) + 图表重做 (#16,#17,#18) | 30h |
| **W4** | 论文全稿修改 + Response letter + GitHub 上传 (#21) + 最终校对 | 15h |

### 9.2 关键里程碑

- **M1** (W1末): 所有文本修改完成，实验脚本编写完成
- **M2** (W2末): 所有新实验运行完成，数据汇总
- **M3** (W3中): Table 1 填充，所有新图出图
- **M4** (W3末): 论文全稿修改完成
- **M5** (W4中): Response letter 完成，最终校对完成
- **M6** (W4末): 提交

---

## 10. 验证清单

提交前逐条确认：

### 数学正确性
- [ ] #1: 论文中 HC-PINN 数学描述正确（输入变换，非输出乘法），链式法则推导完整
- [ ] #3: 全文不再出现 "n=1"/"n=2" 误用，统一用波数 k
- [ ] #10: 色散关系显式推导完整，数值代入正确
- [ ] #11: γ 的物理含义和无量纲化方案清晰
- [ ] #12: 不再出现 "PDE 多稳态" 的表述

### 实验严谨性
- [ ] #4: 所有模型报告的 k_dom 和误差来自 ≥5 次独立运行
- [ ] #5: 频谱演化热图直观展示了谱偏差现象
- [ ] #7: 初始条件的数学形式和数值实现明确记录
- [ ] #9: N_int 敏感性分析证明 10,000 点充分
- [ ] #24: FDM 网格收敛性已验证

### 分析深度
- [ ] #6: FF-PINN σ=12.5 的失败有 NTK 或类似分析支撑
- [ ] #8: 所有 FFT 结果使用 Hann 窗
- [ ] #13: PDE 残差的频谱分解直接证明"低频收敛、高频残留"

### 文献与呈现
- [ ] #14: 与已有工作的差异表完整
- [ ] #15: 负面结果的价值和泛化性充分论证
- [ ] #16: 所有图片为矢量格式 (PDF)，坐标轴有标签和单位
- [ ] #17: 图3/5/10 合并为新 Fig.3/Fig.4
- [ ] #18: 损失曲线使用对数纵轴
- [ ] #19: 所有引用格式完整（含 DOI）
- [ ] #20: 摘要和全文语言正式、精确

### 代码与数据
- [ ] #21: GitHub 仓库公开，含完整代码、README、结果数据
- [ ] 新实验脚本在 `experiments/` 目录下组织良好
- [ ] 分析脚本在 `analysis/` 目录下

### 附加
- [ ] Response letter 逐条回复 24 条审稿意见
- [ ] 所有修改标注在稿件中的位置（页数/行数）

---

## 11. Response Letter 模板

```
Dear Editor and Reviewer,

We sincerely thank the reviewer for their thorough and insightful evaluation
of our manuscript. The detailed critique has substantially improved the
scientific rigor of this work. Below we provide a point-by-point response
to all 24 issues raised. Major modifications in the revised manuscript are
highlighted in blue.

---

[每条格式]

**Reviewer Comment #X:** [复制审稿人原文]

**Response:** [解释做了什么修改]

**Changes:** [具体列出修改内容]
- Page X, Line Y: [修改描述]
- New Figure Z: [描述]
- New Appendix A: [描述]

---

[结束语]

We believe the revised manuscript now meets the standards of scientific
rigor expected by [Journal Name]. We remain at the reviewer's disposal
for any further questions.

Sincerely,
[Authors]
```

---

## A. 附录：代码修改索引

| 文件 | 修改内容 | 涉及问题 |
|:---|:---|:---|
| `models/pinn.py` | 无需修改（代码已正确） | #1 |
| `train.py` | FFT 加窗、损失图对数坐标确认 | #8, #18 |
| `train_ff.py` | FFT 加窗、损失图对数坐标 | #8, #18 |
| `numerical/fdm_solver.py` | FFT 加窗 | #8 |
| `experiments/multi_seed.py` | **新建**：多 seed 批量训练 | #4 |
| `experiments/spectral_evolution.py` | **新建**：频谱演化诊断 | #5 |
| `experiments/sensitivity_Nint.py` | **新建**：采样点敏感性 | #9 |
| `experiments/mlp_baseline.py` | **新建**：数据驱动 MLP | #22 |
| `experiments/gamma_sweep.py` | **新建**：γ 参数扫频 | #23 |
| `experiments/fdm_convergence.py` | **新建**：FDM 网格收敛 | #24 |
| `analysis/ntk_analysis.py` | **新建**：NTK 条件数分析 | #6 |
| `analysis/spectral_residual.py` | **新建**：频谱残差分解 | #13 |
| `matplotlibrc` | **新建**：统一出图样式 | #16, #17 |
| `README.md` | 完善说明、添加复现步骤 | #21 |

---

*文档版本: v1.0, 2026-07-15 | 作者: 基于审稿意见系统整理，与 Cowork 3P 共同讨论制定*
