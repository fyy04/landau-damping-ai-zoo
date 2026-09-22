# Landau Damping AI Zoo

面向 Landau damping 人工智能建模方法的文献、模型与代码整理仓库。

本仓库首先回答三个问题：

1. AI 替代了 Landau damping 数值计算中的哪一部分？
2. 不同论文的输入、输出、训练方式和验证范围有什么区别？
3. 哪些方法只有论文结果，哪些已有公开代码，哪些已经完成复现和统一评测？

当前阶段以研究路线梳理和代码索引为主，尚未形成可直接排名的统一 benchmark。

## 1. 物理问题
本仓库主要关注无碰撞静电等离子体中的 Landau damping。典型的 1D1V Vlasov-Poisson 方程：

$$
\frac{\partial f}{\partial t} + v\frac{\partial f}{\partial x} - E\frac{\partial f}{\partial v} = 0
$$

$$
\frac{\partial E}{\partial x} = 1-n,\quad n(x,t)=\int f(x,v,t)\,dv
$$

其中 $f(x,v,t)$ 是电子相空间分布函数。通过对速度积分，可以得到密度、流速、压力、热流：

$$
n=\int f\,dv,\quad u=\frac{1}{n}\int v f\,dv
$$

$$
p=\int (v-u)^2 f\,dv,\quad q=\int (v-u)^3 f\,dv
$$

低阶流体方程的压力演化需要热流梯度：

$$
g=\frac{\partial q}{\partial x}
$$

但有限个低阶矩通常不能唯一确定更高阶矩，这就是**流体闭合问题**。



常见初始条件为

$$
f(x,v,0) = \left[1+\alpha\cos(kx)\right] \frac{\exp(-v^2/2)}{\sqrt{2\pi}}
$$

其中 $k$ 控制空间尺度、相速度和阻尼行为，$\alpha$ 控制初始扰动强度及非线性程度。


## 2. 研究路线

本仓库按照模型在物理计算链条中替代的位置，将现有工作分为三类。

### 2.1 闭合推演

闭合模型保留低阶流体方程，由 AI 预测缺失的高阶项，再通过数值积分器推进：

$$
[n,u,p,E]_{\mathrm{current/history}}
\xrightarrow{\text{AI closure}}
\widehat{\partial_x q}
\xrightarrow{\text{fluid solver}}
[n,u,p,E]_{t+\Delta t}.
$$

主要子类包括：

- **显式方程发现**：从数据中识别矩方程和可解释闭合公式；
- **PINN 隐式闭合**：利用矩方程残差约束神经时空解；
- **瞬时监督闭合**：由当前低阶矩预测当前热流或热流梯度；
- **历史闭合**：利用一段低阶状态历史补偿被截断的动力学信息；
- **在线/后验有效闭合**：在可微流体求解器中直接优化闭环轨迹；
- **混合闭合**：保留 Hammett-Perkins 等传统闭合，仅学习非线性残差。

闭合模型必须区分两种评价：

- **离线闭合误差**：在动力学真值状态上预测  $q$ 或 $\partial_x q$ 的误差；
- **闭环轨迹误差**：模型接入流体方程后，自主推进得到的状态误差和稳定性。

较低的离线误差不保证长期闭环稳定。

### 2.2 直接代理

直接代理学习参数、初态或当前状态到目标输出的映射，不一定显式求解原来的流体闭合问题。

主要子类包括：

- **积分诊断量代理**：预测电场能量、阻尼率、振荡频率或饱和值；
- **低阶空间场代理**：预测 $(n,u,p,E)$ 等矩场；
- **完整相空间代理**：预测$f(x,v,t)$ 或 $\Delta f(x,v,t)$；
- **Snapshot**：根据参数和时间直接查询单个时刻；
- **Stepper/Rollout**：递归预测下一状态；
- **全轨迹算子**：由初态一次生成完整时空轨迹。

需要注意，“直接代理”不等于“一定没有时间推进”。Snapshot 和全轨迹算子可以不递归积分，而 Stepper、Neural ODE 和宏步代理仍然需要递归或数值积分。

### 2.3 神经增强数值组件

这类方法不替代完整动力学或流体系统，只替换传统求解器中的昂贵组件，例如：

- 神经碰撞算子；
- 神经 score estimator；
- PIC 数据去噪；
- 粗网格误差修正；
- 神经 Poisson 求解器；
- 相空间超分辨率。

这些方法应与相同数值任务下的传统组件比较，不能与完整代理模型混合排名。

## 3. 已阅读文献

| 工作 | 路线 | 输入 | 输出 | 数据/模型 | 是否闭环推进 | 当前证据边界 |
|---|---|---|---|---|---|---|
| Cheng et al., *Data-driven, multi-moment fluid modeling of Landau damping* | 显式方程发现 | 低阶矩及候选导数项 | 多矩 PDE 与显式热流闭合 | Gkeyll 1D1V；mPDE-Net | 是 | 主要基于单一参数案例和窄时空采样，候选项带有较强先验 |
| Wei et al., *Data-Driven Modeling of Landau Damping by Fourier Neural Operator* | 离线监督闭合 | 电子密度等低阶场 | 热流 \(q\) | Gkeyll 1D1V；FNO/MLP | 否 | 报告 FNO 离线误差优于 MLP，但未验证长期流体闭环 |
| Qin et al., *Data-driven modeling of Landau damping by physics-informed neural networks* | PINN 隐式闭合 | 坐标 \((x,t)\)、稀疏观测和初边值 | \(n,u,p,q,E\) | PINN、gPINN、gPINNp | 不属于通用闭合部署 | 主要重建同一初边值问题，不能直接视为跨参数闭合泛化 |
| Huang et al., *Machine-learning heat flux closure for multi-moment fluid modeling of nonlinear Landau damping* | 瞬时 FNO 闭合 | 当前 \([n,u,p](x)\) | \(\partial_xq(x)\) | Gkeyll 1D1V；FNO | 是 | 能复现非线性回弹和 bounce，但训练参数范围和独立案例数量有限 |
| Shekarpaz et al., *Surrogate Modeling of Landau Damping with Deep Operator Networks* | 积分诊断量代理 | 温度参数和时间 | 电场能量曲线 | Gkeyll；DeepONet | 否 | 只输出积分诊断量，不能评价空间场、相位、守恒或完整分布 |
| Liu et al., *Data-driven modeling of electrostatic turbulence by physics-informed Fourier neural operator* | 完整低阶场代理 | 稀疏初态、多场和坐标 | 12 个矩及电场的时空轨迹 | Gkeyll 2D2V；PIFNO | 直接生成轨迹 | 物理约束改善预测，但不等同于逐步闭环或严格守恒 |
| Burles et al., *A Neural Operator Closure for Landau Damping in Electrostatic Plasma* | 历史在线有效闭合 | \([n,u,p,E,\partial_xq]\) 历史和幅值 | 下一步 \(\partial_xq\) | FNO + 可微流体求解器 | 是 | 固定波数、主要扫描幅值；输出是求解器相关的有效闭合；目前为预印本 |
| Ilin & Hu, *A Neural Score-Based Particle Method for the Vlasov-Maxwell-Landau System* | 神经增强数值组件 | 粒子分布样本 | 速度 score \(\nabla_v\log f\) | 在线 score matching | 仍运行粒子求解器 | 研究有碰撞 Landau 算子，不是无碰撞 Landau damping 的完整代理 |
| Finn et al., *A Numerical Study of Landau Damping with PETSc-PIC* | 非 AI 数值基准 | PIC 初态和数值参数 | Vlasov-Poisson 数值解 | PETSc-PIC | 是 | 不属于 AI，但可用于阻尼率、频率、守恒和数值收敛验证 |

## 4. 模型状态

本仓库采用以下状态区分文献整理和代码复现：

| 状态 | 含义 |
|---|---|
| `catalogued` | 已整理论文、任务和实验信息 |
| `upstream-available` | 已找到作者公开代码，但尚未验证 |
| `adapted` | 已接入本仓库的统一接口 |
| `reproduced` | 已复现论文中的关键结果 |
| `benchmarked` | 已在统一数据、划分和指标下完成评测 |

当前状态：

| 模型 | 分类状态 | 代码状态 | 复现状态 |
|---|---|---|---|
| mPDE-Net | `catalogued` | 待核对官方实现和许可证 | 未复现 |
| FNO heat-flux surrogate | `catalogued` | 待核对 | 未复现 |
| PINN/gPINN closure | `catalogued` | 待核对 | 未复现 |
| Nonlinear FNO closure | `catalogued` | 待核对 | 未复现 |
| DeepONet diagnostic surrogate | `catalogued` | 待核对 | 未复现 |
| PIFNO turbulence surrogate | `catalogued` | 待核对 | 未复现 |
| Online memory FNO closure | `catalogued` | 待核对 | 未复现 |
| Neural score particle method | `catalogued` | 论文报告公开实现，待验证 | 未复现 |
| PETSc-PIC baseline | `catalogued` | PETSc 相关实现待验证 | 未复现 |

“论文中报告成功”不等于“本仓库已经复现”。

## 5. 计划收录的代码

第一阶段计划收录或实现以下内容：

- 零热流闭合；
- Hammett-Perkins 闭合；
- Oracle 真值热流闭合；
- MLP/CNN/FNO 离线闭合基线；
- 瞬时 FNO 热流梯度闭合；
- 历史 FNO 闭合；
- HP + 神经残差闭合；
- Snapshot 分布代理；
- Stepper/Rollout 分布代理；
- DeepONet 电场能量代理；
- 统一物理诊断与结果格式。

外部论文代码优先通过链接、固定 commit、submodule 或 adapter 管理。许可证不明确的代码不会直接复制到本仓库。

大型数据、模型权重、论文 PDF 和完整实验输出不直接提交到 Git。

## 6. Benchmark 发展路线

### 阶段一：文献与代码目录

- 整理研究路线和术语；
- 为论文建立统一 model card；
- 核对官方代码、commit、依赖和许可证；
- 区分论文结果、代码可用和本地复现状态。

### 阶段二：统一数据合同

- 固定变量定义、单位和数组形状；
- 按完整物理 case 划分数据；
- 归一化只使用训练集；
- 保存 $(k,\alpha)$、时间坐标、求解器配置和数据版本；
- 对参考解执行守恒、正性和分辨率检查。

### 阶段三：闭合 benchmark

统一比较：

- 零热流；
- Hammett-Perkins；
- 瞬时神经闭合；
- 历史神经闭合；
- HP + 神经残差；
- 离线监督、混合训练和在线有效闭合。

评价闭合标签误差、完整轨迹误差、稳定性、守恒、场能、相位和端到端时间。

### 阶段四：直接代理 benchmark

分别建立：

- 积分诊断量榜；
- 低阶空间场榜；
- 完整相空间 Snapshot 榜；
- 自回归 Rollout 榜。

不同输出任务不混合排名。

### 阶段五：泛化与盲测

测试范围包括：

- 参数域内插值；
- $k$ 外推；
- $\alpha$ 外推；
- 时间外推；
- 分辨率和时间步迁移；
- 跨求解器测试；
- 模型、阈值和代码冻结后的独立 holdout。

## 7. 评价原则

统一 benchmark 至少应遵守：

1. 数据按完整 case 划分，不随机拆分相邻时间帧；
2. 区分离线标签误差和自主闭环误差；
3. 先报告完成率、非有限值、负密度和负压力；
4. 再报告场能、第一模幅相、扰动相对误差和守恒漂移；
5. 失败案例不能从平均指标中静默删除；
6. 比较两个模型时使用相同案例和时间区间；
7. 推理时间必须注明硬件、批量、输出内容、加载、传输和写盘范围；
8. 论文原始结果与本仓库统一复现结果分开保存。

## 8. 当前限制

本仓库目前处于资料整理阶段：

- 尚未形成统一训练数据；
- 尚未验证全部论文代码；
- 尚未完成相同条件下的模型复现；
- 尚未建立正式排行榜；
- 文献使用的物理范围、网格、输出和指标不同，论文数字不能直接横向排名。

欢迎提交文献补充、代码来源、复现记录和可验证的实验配置。
````
