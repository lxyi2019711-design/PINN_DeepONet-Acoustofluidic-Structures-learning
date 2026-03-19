"""
loss.py — 物理约束损失函数（v5 声-流-固耦合版）
================================================================
【v5 升级说明】
  1. 输出变更为 (u, v, p)，物理损失可使用完整 Navier-Stokes 方程
  2. 新增 ns_momentum_loss(): 含压力梯度的 Stokes 方程残差
  3. 保留连续性约束、自适应权重、热粘性边界损失接口
  4. 保留 warmup 机制（训练初期禁用高阶导数项）

═══════════════════════════════════════════════════════════════════
PINN（Physics-Informed Neural Network）物理约束原理
═══════════════════════════════════════════════════════════════════

PINN 的核心思想：将控制方程的残差作为损失函数的一部分，
使神经网络的预测不仅拟合数据，还满足物理规律。

总损失 L_total = L_data + λ_pde · L_pde

其中 L_pde 包含以下物理约束：

1. 连续性方程（质量守恒）：
   ∂u/∂x + ∂v/∂y = 0
   含义：不可压缩流体的速度场必须无散度。

2. Stokes 方程（低 Reynolds 数动量守恒）：
   -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²) + F_x = 0  （x 方向）
   -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²) + F_y = 0  （y 方向）
   其中 F_x, F_y 为声致体力（Lighthill 应力梯度），此处简化为零。

   物理背景：声流是由声波衰减产生的二阶效应（Reynolds stress），
   其 Reynolds 数通常很小（Re << 1），因此惯性项可忽略，
   采用 Stokes 流近似。

3. 边界条件（固体表面无滑移）：
   u|_wall = 0, v|_wall = 0

损失函数的数学意义：
  L_pde = (1/N) Σᵢ [R(xᵢ, yᵢ)]²
  其中 R 为 PDE 残差，N 为配置点数。
  L_pde → 0 意味着预测场在配置点处满足 PDE。

多物理场损失的梯度病态问题：
  速度量级 ~ O(10⁻⁵) m/s，压力量级 ~ O(10⁴) Pa，
  导致各损失项梯度尺度差异巨大。
  解决方案：自适应权重（AdaptiveLossWeights）+ 特征归一化
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import torch                                     # PyTorch
import torch.nn as nn                            # 神经网络模块
from config import params, device                # 物理参数和设备


# ═══════════════════════════════════════════════════════════════════
# 自适应损失权重（基于不确定性加权）
# ═══════════════════════════════════════════════════════════════════
class AdaptiveLossWeights(nn.Module):
    """
    自适应多任务损失权重（Kendall et al., CVPR 2018）。

    原理：
      每个损失项 L_i 对应一个可学习的对数方差参数 log(σ²_i)。
      加权损失 = Σᵢ [ L_i / (2σ²_i) + log(σ_i) ]
      σ_i 通过反向传播自动调整：
        - 误差大的任务 → σ_i 增大 → 权重 1/σ² 减小（降低该任务权重）
        - 误差小的任务 → σ_i 减小 → 权重 1/σ² 增大（提升该任务权重）

    优势：
      自动平衡量纲差异大的多物理场损失，无需手动调节 λ 参数。

    参数:
        n_tasks: 损失项数量（默认 2：数据损失 + PDE 损失）
    """

    def __init__(self, n_tasks: int = 2):
        super().__init__()
        # log(σ²)，初始化为 0 → σ = 1，各任务初始权重相同
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses) -> tuple:
        """
        计算加权总损失。

        参数:
            *losses: n_tasks 个标量 loss
        返回:
            (total_loss, weights_list)
        """
        assert len(losses) == len(self.log_vars)   # 检查损失项数量
        total = torch.zeros(1, device=self.log_vars.device)
        weights = []
        for loss_i, lv in zip(losses, self.log_vars):
            w = torch.exp(-lv)                     # 精度权重 = 1/σ²
            total = total + w * loss_i + lv        # 加权损失 + 正则项
            weights.append(w.item())
        return total.squeeze(), weights

    def get_weights(self) -> list:
        """返回当前各任务权重（用于日志）。"""
        return [torch.exp(-lv).item() for lv in self.log_vars]


# ═══════════════════════════════════════════════════════════════════
# 物理损失 1：连续性方程（不可压缩条件）
# ═══════════════════════════════════════════════════════════════════
def continuity_loss(model: nn.Module,
                    x_col: torch.Tensor,
                    y_norm_obj) -> torch.Tensor:
    """
    不可压缩连续性方程残差：∂u/∂x + ∂v/∂y = 0

    数学推导：
      对于不可压缩流体（∇·u = 0），速度场的散度为零。
      在二维情况下：∂u/∂x + ∂v/∂y = 0
      残差 R_cont = ∂u/∂x + ∂v/∂y
      损失 L_cont = (1/N) Σ R_cont²

    实现细节：
      - 梯度通过 autograd 在归一化坐标系下计算
      - 需要先反归一化预测值到真实物理量级
      - create_graph=True 允许高阶梯度（用于粘性项）

    参数:
        model:      神经网络模型
        x_col:      配置点（归一化输入），(N, 3)
        y_norm_obj: 输出归一化器（用于反归一化）
    返回:
        标量 tensor，连续性方程均方残差
    """
    x_c     = x_col.clone().requires_grad_(True)   # 需要梯度的输入副本
    out     = model(x_c)                            # 模型预测（归一化输出）
    out_real = y_norm_obj.inverse_transform(out)    # 反归一化到真实物理量

    u_pred = out_real[:, 0:1]                       # u 速度分量
    v_pred = out_real[:, 1:2]                       # v 速度分量

    # 计算一阶偏导数
    def grad1(f, col):
        """计算 ∂f/∂x_col[col]，其中 col 指定对哪个输入列求导。"""
        return torch.autograd.grad(
            f.sum(), x_c, create_graph=True
        )[0][:, col:col+1]

    u_x = grad1(u_pred, 0)                         # ∂u/∂x
    v_y = grad1(v_pred, 1)                          # ∂v/∂y

    # 连续性残差的均方值
    return torch.mean((u_x + v_y) ** 2)


# ═══════════════════════════════════════════════════════════════════
# 物理损失 2：Stokes 方程动量守恒（含压力梯度）
# ═══════════════════════════════════════════════════════════════════
def ns_momentum_loss(model: nn.Module,
                     x_col: torch.Tensor,
                     y_norm_obj) -> torch.Tensor:
    """
    Stokes 方程动量残差（低 Re 声流近似，含压力梯度项）。

    数学形式（Stokes 方程，忽略惯性项和声致体力）：
      x 方向：-∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²) = 0
      y 方向：-∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²) = 0

    残差：
      R_x = -∂p/∂x + μ∇²u
      R_y = -∂p/∂y + μ∇²v
      L_mom = (1/N) Σ (R_x² + R_y²)

    物理说明：
      声流 Reynolds 数 Re = ρUL/μ 通常 << 1（U ~ 10⁻⁴ m/s, L ~ 10⁻⁴ m），
      因此可忽略非线性惯性项 ρ(u·∇)u，采用 Stokes 流近似。
      声致体力（Lighthill 应力梯度）在此简化为零——
      该力已隐含在训练数据中，模型通过数据拟合间接学习其效应。

    参数:
        model:      神经网络模型
        x_col:      配置点（归一化输入），(N, 3)
        y_norm_obj: 输出归一化器
    返回:
        标量 tensor，Stokes 方程均方残差
    """
    mu = params.mu                                  # 动力粘度

    x_c      = x_col.clone().requires_grad_(True)
    out      = model(x_c)
    out_real = y_norm_obj.inverse_transform(out)

    u_pred = out_real[:, 0:1]                       # u 速度
    v_pred = out_real[:, 1:2]                       # v 速度
    p_pred = out_real[:, 2:3]                       # p 压力

    def grad1(f, col):
        """一阶偏导 ∂f/∂x_col[col]"""
        return torch.autograd.grad(
            f.sum(), x_c, create_graph=True, retain_graph=True
        )[0][:, col:col+1]

    def grad2(f, col):
        """二阶偏导 ∂²f/∂x_col[col]²"""
        f1 = grad1(f, col)                          # 先求一阶导
        return torch.autograd.grad(
            f1.sum(), x_c, create_graph=True, retain_graph=True
        )[0][:, col:col+1]                          # 再对同一变量求导

    # 压力梯度
    p_x = grad1(p_pred, 0)                          # ∂p/∂x
    p_y = grad1(p_pred, 1)                          # ∂p/∂y

    # 速度 Laplacian（粘性项）
    u_xx = grad2(u_pred, 0)                         # ∂²u/∂x²
    u_yy = grad2(u_pred, 1)                         # ∂²u/∂y²
    v_xx = grad2(v_pred, 0)                         # ∂²v/∂x²
    v_yy = grad2(v_pred, 1)                         # ∂²v/∂y²

    # Stokes 方程残差
    R_x = -p_x + mu * (u_xx + u_yy)                # x 方向动量残差
    R_y = -p_y + mu * (v_xx + v_yy)                # y 方向动量残差

    return torch.mean(R_x ** 2) + torch.mean(R_y ** 2)


# ═══════════════════════════════════════════════════════════════════
# 物理损失 3：边界无滑移条件
# ═══════════════════════════════════════════════════════════════════
def boundary_noslip_loss(model: nn.Module,
                         x_boundary: torch.Tensor,
                         y_norm_obj) -> torch.Tensor:
    """
    固体表面无滑移边界条件：u|_wall = 0, v|_wall = 0。

    物理含义：
      在粘性流体中，流体在固体壁面处的速度等于壁面速度。
      对于静止固体壁面：u = 0, v = 0。

    参数:
        x_boundary: 边界点坐标（归一化输入），(N_b, 3)
        y_norm_obj: 输出归一化器
    返回:
        标量 tensor，边界条件残差
    """
    if x_boundary is None or len(x_boundary) == 0:
        return torch.tensor(0.0, device=device)

    out      = model(x_boundary)
    out_real = y_norm_obj.inverse_transform(out)
    u_bc     = out_real[:, 0]                       # 边界处 u（应为 0）
    v_bc     = out_real[:, 1]                       # 边界处 v（应为 0）

    return torch.mean(u_bc ** 2) + torch.mean(v_bc ** 2)


# ═══════════════════════════════════════════════════════════════════
# 总损失函数
# ═══════════════════════════════════════════════════════════════════
def total_loss(model: nn.Module,
               X_batch: torch.Tensor,
               Y_batch: torch.Tensor,
               x_norm_obj,
               y_norm_obj,
               adaptive_weights=None,
               lambda_data:       float = 1.0,
               lambda_continuity: float = 0.01,
               lambda_momentum:   float = 0.001,
               lambda_boundary:   float = 0.0,
               x_boundary:        torch.Tensor = None,
               use_momentum:      bool  = True) -> tuple:
    """
    计算总损失 = 数据损失 + 物理约束损失。

    损失组成：
      L_total = w_data · L_data + w_pde · L_pde + λ_bc · L_bc

    其中：
      L_data = MSE(Y_pred, Y_true)         — 数据拟合损失
      L_pde  = λ_cont · L_cont + λ_mom · L_mom  — PDE 约束损失
        L_cont = 连续性方程残差
        L_mom  = Stokes 动量方程残差（含压力梯度 + 粘性项）
      L_bc   = 边界无滑移残差

    参数:
        model:             网络模型
        X_batch, Y_batch:  输入/输出批次数据（标准化后）
        x_norm_obj:        输入归一化器
        y_norm_obj:        输出归一化器
        adaptive_weights:  自适应权重实例（可选）
        lambda_data:       数据损失系数
        lambda_continuity: 连续性损失系数
        lambda_momentum:   动量方程损失系数
        lambda_boundary:   边界条件损失系数
        x_boundary:        边界采样点（可选）
        use_momentum:      是否启用动量方程（训练初期可禁用）
    返回:
        (loss_total, loss_dict) — 总损失标量和各项详细分解
    """
    # 1. 数据拟合损失（MSE）
    Y_pred = model(X_batch)
    loss_data = nn.MSELoss()(Y_pred, Y_batch)

    # 2. 连续性方程残差（取少量配置点，控制计算开销）
    n_col     = min(128, len(X_batch))              # 配置点数（最多 128）
    loss_cont = continuity_loss(model, X_batch[:n_col], y_norm_obj)

    # 3. Stokes 动量方程残差（二阶导计算成本较高）
    if use_momentum:
        n_mom     = min(64, len(X_batch))           # 动量方程配置点（最多 64）
        loss_mom  = ns_momentum_loss(model, X_batch[:n_mom], y_norm_obj)
    else:
        loss_mom  = torch.tensor(0.0, device=device)  # warmup 期间跳过

    # 4. 边界无滑移条件
    loss_bc = boundary_noslip_loss(model, x_boundary, y_norm_obj)

    # 合并 PDE 损失
    loss_pde = lambda_continuity * loss_cont + lambda_momentum * loss_mom

    # 自适应加权 vs 手动加权
    if adaptive_weights is not None:
        loss_total, aw = adaptive_weights(loss_data, loss_pde)
        loss_total = loss_total + lambda_boundary * loss_bc
        w_data, w_pde = aw
    else:
        loss_total = (lambda_data * loss_data +
                      loss_pde +
                      lambda_boundary * loss_bc)
        w_data, w_pde = lambda_data, 1.0

    # 返回总损失和各项分解
    return loss_total, {
        'total'       : loss_total.item(),
        'fluid_data'  : loss_data.item(),
        'continuity'  : loss_cont.item(),
        'momentum'    : loss_mom.item(),
        'boundary_bc' : loss_bc.item(),
        'w_data'      : w_data,
        'w_pde'       : w_pde,
    }


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from model import AcousticStreamingPINN
    from data_loader import Normalizer

    model = AcousticStreamingPINN().to(device)

    N = 32
    X_fake = torch.rand(N, 3).to(device)
    Y_fake = torch.rand(N, 3).to(device)

    def _make_norm(t):
        n = Normalizer()
        n.mean = t.mean(dim=0, keepdim=True)
        n.std  = t.std(dim=0, keepdim=True)
        n.std[n.std < 1e-8] = 1.0
        return n

    xn = _make_norm(X_fake)
    yn = _make_norm(Y_fake)

    # 测试含动量方程的损失
    loss, ld = total_loss(model, xn.transform(X_fake), yn.transform(Y_fake),
                          xn, yn, use_momentum=True)
    print("完整损失:", {k: f"{v:.4e}" for k, v in ld.items()})

    # 测试自适应权重
    aw = AdaptiveLossWeights(n_tasks=2).to(device)
    loss2, ld2 = total_loss(model, xn.transform(X_fake), yn.transform(Y_fake),
                            xn, yn, adaptive_weights=aw, use_momentum=False)
    print("自适应权重:", {k: f"{v:.4e}" for k, v in ld2.items()})
    print("[loss v5] 自检通过 ✅")
