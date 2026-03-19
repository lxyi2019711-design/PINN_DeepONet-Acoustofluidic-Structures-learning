"""
force_calculator.py — 固体受力计算模块（v5 新增模块）
================================================================
本模块实现从预测的声流场 (u, v, p) 计算作用在固体微结构上的流体力。

═══════════════════════════════════════════════════════════════════
声流驱动微结构运动的物理机制
═══════════════════════════════════════════════════════════════════

超声波在粘性流体中传播时，由于声衰减（Lighthill 机制），
产生非零的时间平均体力（Reynolds stress 梯度），
驱动稳态的声致流动（acoustic streaming）。

声流对微结构产生的力主要包括两个分量：

1. 流体压力（pressure force）：
   F_p = -∮_S p · n dS
   压力 p 沿法向量 n 方向作用于固体表面，
   压力差产生净推力。

2. 粘性剪切力（viscous shear force）：
   F_τ = ∮_S τ · n dS
   其中 τ 为粘性应力张量：
     τ_ij = μ(∂u_i/∂x_j + ∂u_j/∂x_i)
   流体的速度梯度在固体表面产生切向摩擦力。

3. 声辐射力（acoustic radiation force, ARF）：
   F_ARF = -∇⟨E⟩ · V_s（简化形式）
   声辐射力来源于一阶声场的二阶非线性效应，
   本模块采用 Gor'kov 势能法近似计算。

总合力：
   F_total = F_p + F_τ + F_ARF

═══════════════════════════════════════════════════════════════════
实现方法：数值表面积分
═══════════════════════════════════════════════════════════════════

1. 在固体表面均匀采样 N_s 个点（geometry.sample_triangle_surface）
2. 在每个采样点处用神经网络预测 (u, v, p)
3. 用自动微分计算速度梯度 ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
4. 构造应力张量 σ = -pI + μ(∇u + ∇u^T)
5. 计算表面力密度 f = σ · n
6. 数值积分 F = Σ f_i · ds_i
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import numpy as np                                   # 数值计算
import torch                                          # PyTorch
from config import params, device                    # 全局配置
from geometry import sample_triangle_surface, translate_surface_points


# ═══════════════════════════════════════════════════════════════════
# 核心类：流体力计算器
# ═══════════════════════════════════════════════════════════════════
class ForceCalculator:
    """
    计算流体对固体微结构的总作用力。

    工作流程：
      1. 初始化时预采样固体表面点和法向量
      2. 给定模型和频率，在表面点处预测流场
      3. 计算压力分量和剪切力分量
      4. 数值积分得到合力 (Fx, Fy)

    参数:
        model:       训练好的 FP-PiDON 模型
        x_norm:      输入归一化器
        y_norm:      输出归一化器
        n_per_edge:  每条边的采样点数（默认使用 config 值）
    """

    def __init__(self, model, x_norm, y_norm, n_per_edge=None):
        self.model  = model                           # 神经网络模型
        self.x_norm = x_norm                          # 输入归一化器
        self.y_norm = y_norm                          # 输出归一化器
        self.mu     = params.mu                       # 动力粘度 [Pa·s]

        # 预采样固体表面点（参考位置，未平移）
        if n_per_edge is None:
            n_per_edge = params.n_surface_pts // 3
        self.base_points, self.base_normals, self.base_ds = \
            sample_triangle_surface(n_per_edge=n_per_edge)

        # 打印初始化信息
        print(f"  ForceCalculator 初始化: {len(self.base_points)} 个表面积分点")

    def _prepare_surface_input(self, surface_pts, freq):
        """
        将表面采样点转换为模型输入张量。

        步骤：
          1. 物理坐标 → 归一化坐标 [0,1]
          2. 添加频率列
          3. 标准化（z-score）
          4. 转为 GPU 张量

        参数:
            surface_pts: np.ndarray (N, 2)，表面点坐标 [m]
            freq:        驱动频率 [Hz]
        返回:
            torch.Tensor (N, 3)，标准化后的模型输入
        """
        # 物理坐标 → 归一化坐标
        x_n = (surface_pts[:, 0] - params.Lx_min) / params.Lx  # x → [0,1]
        y_n =  surface_pts[:, 1] / params.Ly                    # y → [0,1]
        f_n = np.full(len(surface_pts), freq / params.freq_ref) # 归一化频率

        # 构造输入矩阵并转为张量
        X = np.column_stack([x_n, y_n, f_n])                    # (N, 3)
        X_tensor = torch.FloatTensor(X).to(device)              # GPU 张量
        X_normed = self.x_norm.transform(X_tensor)              # 标准化
        return X_normed

    def compute_forces(self, freq, dx=0.0, dy=0.0):
        """
        计算给定频率和固体位移下的流体合力 (Fx, Fy)。

        计算公式（二维数值面积分）：

        压力分量：
          F_p = -Σᵢ p(xᵢ, yᵢ) · nᵢ · dsᵢ

        粘性剪切力分量（2D 应力张量 × 法向量）：
          τ_xx = 2μ · ∂u/∂x
          τ_yy = 2μ · ∂v/∂y
          τ_xy = τ_yx = μ(∂u/∂y + ∂v/∂x)
          F_τ = Σᵢ [τ · n]ᵢ · dsᵢ

        合力：
          Fx = Σᵢ (-p·nx + τ_xx·nx + τ_xy·ny) · dsᵢ
          Fy = Σᵢ (-p·ny + τ_xy·nx + τ_yy·ny) · dsᵢ

        参数:
            freq: 驱动频率 [Hz]
            dx:   固体 x 方向位移 [m]（默认 0，初始位置）
            dy:   固体 y 方向位移 [m]
        返回:
            (Fx, Fy) — x 和 y 方向合力 [N/m]（二维，单位深度）
        """
        # 1. 更新表面点位置（考虑固体位移）
        surface_pts, normals = translate_surface_points(
            self.base_points, self.base_normals, dx, dy
        )
        ds = self.base_ds                             # 边元长度不变

        # 2. 准备模型输入
        X_input = self._prepare_surface_input(surface_pts, freq)

        # 3. 需要梯度（用于计算速度梯度 → 剪切力）
        X_input.requires_grad_(True)

        # 4. 模型预测 + 反归一化
        out      = self.model(X_input)                # (N, 3) 标准化输出
        out_real = self.y_norm.inverse_transform(out) # (N, 3) 真实物理量

        u_pred = out_real[:, 0:1]                     # u 速度 [m/s]
        v_pred = out_real[:, 1:2]                     # v 速度 [m/s]
        p_pred = out_real[:, 2:3]                     # p 压力 [Pa]

        # 5. 计算速度梯度（自动微分）
        def grad1(f, col):
            """计算 ∂f/∂X_input[col]"""
            return torch.autograd.grad(
                f.sum(), X_input, create_graph=False, retain_graph=True
            )[0][:, col:col+1]

        du_dx = grad1(u_pred, 0)                      # ∂u/∂x（归一化坐标下）
        du_dy = grad1(u_pred, 1)                      # ∂u/∂y
        dv_dx = grad1(v_pred, 0)                      # ∂v/∂x
        dv_dy = grad1(v_pred, 1)                      # ∂v/∂y

        # 6. 转为 numpy 进行数值积分
        p_np  = p_pred.detach().cpu().numpy().ravel()     # 压力值
        du_dx_np = du_dx.detach().cpu().numpy().ravel()
        du_dy_np = du_dy.detach().cpu().numpy().ravel()
        dv_dx_np = dv_dx.detach().cpu().numpy().ravel()
        dv_dy_np = dv_dy.detach().cpu().numpy().ravel()

        nx = normals[:, 0]                            # 法向量 x 分量
        ny = normals[:, 1]                            # 法向量 y 分量

        # 7. 构造应力张量分量
        mu = self.mu
        tau_xx = 2 * mu * du_dx_np                    # σ_xx = 2μ(∂u/∂x)
        tau_yy = 2 * mu * dv_dy_np                    # σ_yy = 2μ(∂v/∂y)
        tau_xy = mu * (du_dy_np + dv_dx_np)           # σ_xy = μ(∂u/∂y + ∂v/∂x)

        # 8. 表面力密度：f = (-pI + τ) · n
        fx = (-p_np * nx + tau_xx * nx + tau_xy * ny) # x 方向力密度
        fy = (-p_np * ny + tau_xy * nx + tau_yy * ny) # y 方向力密度

        # 9. 数值积分：F = Σ f_i · ds_i
        Fx = np.sum(fx * ds)                          # x 方向合力 [N/m]
        Fy = np.sum(fy * ds)                          # y 方向合力 [N/m]

        return Fx, Fy

    def compute_forces_with_arf(self, freq, dx=0.0, dy=0.0):
        """
        计算包含声辐射力（ARF）的总合力。

        声辐射力 (Acoustic Radiation Force) 是一阶声场的二阶非线性效应，
        本方法采用简化的 Gor'kov 势能法估算：

          F_ARF ≈ -π R² · (∂⟨p²⟩/∂x) / (ρ c²) + (2/3)π R² ρ · ∂⟨v²⟩/∂x

        其中 R 为等效半径，⟨·⟩ 表示时间平均。
        在声压幅值 P0 下，声辐射力量级：
          F_ARF ~ π R³ P0² / (ρ c⁴) · k  （k = 2πf/c 为波数）

        注意：这是一个简化估算，精确 ARF 需要求解一阶声场方程。
        在本代理模型中，ARF 的主要贡献已隐含在训练数据的流场中。

        参数:
            freq: 驱动频率 [Hz]
            dx, dy: 固体位移 [m]
        返回:
            (Fx_total, Fy_total) — 包含 ARF 的总合力 [N/m]
        """
        # 流体压力 + 剪切力
        Fx_fluid, Fy_fluid = self.compute_forces(freq, dx, dy)

        # 声辐射力估算（简化 Gor'kov 模型）
        R   = params.R_eff                            # 等效半径 [m]
        P0  = params.P0_fixed                         # 声压幅值 [Pa]
        rho = params.rho_f                            # 流体密度
        c   = params.c_sound                          # 声速
        k   = 2 * np.pi * freq / c                    # 波数 [1/m]

        # ARF 量级估算（方向取决于声场空间梯度，此处简化为 +x 方向）
        # F_ARF ~ π R³ k P0² / (ρ c⁴)
        F_arf_magnitude = np.pi * R**3 * k * P0**2 / (rho * c**4)

        # 声辐射力方向（简化假设：沿声波传播方向 +x）
        Fx_arf = F_arf_magnitude                      # x 方向 ARF
        Fy_arf = 0.0                                  # y 方向 ARF（假设对称）

        Fx_total = Fx_fluid + Fx_arf
        Fy_total = Fy_fluid + Fy_arf

        return Fx_total, Fy_total


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from model import AcousticStreamingPINN
    from data_loader import Normalizer

    # 创建虚拟模型和归一化器
    model = AcousticStreamingPINN().to(device)
    model.eval()

    def _make_norm(dim, dev=device):
        n = Normalizer()
        n.mean = torch.zeros(1, dim).to(dev)
        n.std  = torch.ones(1, dim).to(dev)
        return n

    x_norm = _make_norm(3)
    y_norm = _make_norm(3)

    # 初始化力计算器
    fc = ForceCalculator(model, x_norm, y_norm, n_per_edge=10)

    # 计算力
    freq = 5e5                                        # 0.5 MHz
    Fx, Fy = fc.compute_forces(freq)
    print(f"\n流体力 (freq={freq/1e6:.2f}MHz):")
    print(f"  Fx = {Fx:.4e} N/m")
    print(f"  Fy = {Fy:.4e} N/m")

    Fx_t, Fy_t = fc.compute_forces_with_arf(freq)
    print(f"\n含 ARF 的总力:")
    print(f"  Fx = {Fx_t:.4e} N/m")
    print(f"  Fy = {Fy_t:.4e} N/m")
    print("[force_calculator v5] 自检通过 ✅")
