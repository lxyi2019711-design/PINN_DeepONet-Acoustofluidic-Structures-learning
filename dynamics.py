"""
dynamics.py — 微结构动力学仿真模块（v5 新增模块）
================================================================
本模块基于牛顿运动方程和时间积分，
预测声流驱动下微结构的低频运动轨迹。

═══════════════════════════════════════════════════════════════════
动力学积分原理
═══════════════════════════════════════════════════════════════════

微结构在声流场中的运动满足牛顿第二定律：

  m · a = F_fluid + F_ARF + F_drag

各力的物理意义：

1. 流体力 F_fluid：
   由流体压力和粘性剪切力的表面积分得到（force_calculator 模块计算）
   F_fluid = ∮_S (-pI + τ) · n dS

2. 声辐射力 F_ARF：
   一阶声场的二阶非线性效应（force_calculator 模块估算）

3. Stokes 阻力 F_drag：
   微结构运动时受到的粘性阻力（反对运动方向）
   F_drag = -γ · v_solid
   其中 γ = 6πμR_eff 为 Stokes 阻力系数，
   v_solid 为固体运动速度。

运动方程（二维）：
  m · d²x/dt² = Fx - γ · dx/dt
  m · d²y/dt² = Fy - γ · dy/dt

状态空间形式（一阶 ODE 系统）：
  令 q = [x, y, vx, vy]ᵀ（位置 + 速度），则：
  dq/dt = [vx, vy, (Fx - γ·vx)/m, (Fy - γ·vy)/m]ᵀ

═══════════════════════════════════════════════════════════════════
时间积分方法
═══════════════════════════════════════════════════════════════════

本模块提供两种时间积分方法：

1. 前向 Euler 法（一阶精度）：
   q(t+Δt) = q(t) + Δt · f(q(t))
   简单快速，但精度较低，适合定性分析。

2. 经典四阶 Runge-Kutta 法（RK4，四阶精度）：
   k1 = f(t, q)
   k2 = f(t+Δt/2, q+Δt/2·k1)
   k3 = f(t+Δt/2, q+Δt/2·k2)
   k4 = f(t+Δt, q+Δt·k3)
   q(t+Δt) = q(t) + (Δt/6)(k1 + 2k2 + 2k3 + k4)
   精度高，稳定性好，适合定量预测。

注：由于 force_calculator 需要调用神经网络和自动微分，
    RK4 每步需要 4 次力评估，计算开销约为 Euler 法的 4 倍。
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import numpy as np                                   # 数值计算
from config import params                            # 物理参数


# ═══════════════════════════════════════════════════════════════════
# 微结构动力学仿真器
# ═══════════════════════════════════════════════════════════════════
class MicrostructureDynamics:
    """
    微结构在声流场中的动力学仿真。

    状态向量 q = [x, y, vx, vy]
      x, y:   微结构质心位置 [m]
      vx, vy: 微结构运动速度 [m/s]

    参数:
        force_calculator: ForceCalculator 实例（计算外力）
        freq:             驱动频率 [Hz]
        m:                固体质量 [kg]（默认使用 config 值）
        drag_coeff:       Stokes 阻力系数 [N·s/m]（默认使用 config 值）
        dt:               时间步长 [s]（默认使用 config 值）
        use_arf:          是否包含声辐射力（默认 True）
    """

    def __init__(self, force_calculator, freq,
                 m=None, drag_coeff=None, dt=None, use_arf=True):
        self.fc         = force_calculator            # 力计算器
        self.freq       = freq                        # 驱动频率 [Hz]
        self.m          = m or params.m_solid          # 质量 [kg]
        self.gamma      = drag_coeff or params.drag_coeff  # 阻力系数 [N·s/m]
        self.dt         = dt or params.dt_dynamics     # 时间步长 [s]
        self.use_arf    = use_arf                      # 是否含 ARF

    def _compute_force(self, dx, dy):
        """
        计算当前位移下的外力。

        参数:
            dx, dy: 固体相对初始位置的位移 [m]
        返回:
            (Fx, Fy) 外力 [N/m]
        """
        if self.use_arf:
            return self.fc.compute_forces_with_arf(self.freq, dx, dy)
        else:
            return self.fc.compute_forces(self.freq, dx, dy)

    def _state_derivative(self, q):
        """
        计算状态向量的时间导数 dq/dt。

        状态方程：
          dx/dt  = vx
          dy/dt  = vy
          dvx/dt = (Fx - γ·vx) / m
          dvy/dt = (Fy - γ·vy) / m

        参数:
            q: [x, y, vx, vy] 状态向量
        返回:
            dq: [vx, vy, ax, ay] 状态导数
        """
        x, y, vx, vy = q                             # 解包状态

        # 计算外力
        Fx, Fy = self._compute_force(x, y)

        # 加速度（牛顿第二定律 + Stokes 阻力）
        ax = (Fx - self.gamma * vx) / self.m          # x 方向加速度
        ay = (Fy - self.gamma * vy) / self.m          # y 方向加速度

        return np.array([vx, vy, ax, ay])

    def simulate_euler(self, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0,
                       n_steps=None):
        """
        前向 Euler 法时间积分。

        递推公式：
          q(t+Δt) = q(t) + Δt · f(q(t))

        优点：实现简单，计算快速
        缺点：一阶精度，大步长时可能不稳定

        参数:
            x0, y0:     初始位置 [m]
            vx0, vy0:   初始速度 [m/s]
            n_steps:    时间步数（默认使用 config 值）
        返回:
            trajectory: np.ndarray (n_steps+1, 5)，各列为 [t, x, y, vx, vy]
        """
        if n_steps is None:
            n_steps = params.n_steps

        dt = self.dt
        q  = np.array([x0, y0, vx0, vy0])            # 初始状态

        # 预分配轨迹数组
        trajectory = np.zeros((n_steps + 1, 5))       # [t, x, y, vx, vy]
        trajectory[0] = [0.0, *q]                     # 初始条件

        print(f"  开始 Euler 积分: {n_steps} 步, dt={dt:.1e}s...")

        for step in range(n_steps):
            dq = self._state_derivative(q)             # 计算导数
            q  = q + dt * dq                           # Euler 更新
            t  = (step + 1) * dt                       # 当前时间

            trajectory[step + 1] = [t, *q]

            # 周期性进度提示
            if (step + 1) % max(1, n_steps // 10) == 0:
                print(f"    步 {step+1}/{n_steps}: "
                      f"x={q[0]*1e6:.2f}μm, y={q[1]*1e6:.2f}μm, "
                      f"|v|={np.sqrt(q[2]**2+q[3]**2)*1e6:.2f}μm/s")

        return trajectory

    def simulate_rk4(self, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0,
                     n_steps=None):
        """
        经典四阶 Runge-Kutta 法（RK4）时间积分。

        递推公式：
          k1 = f(q_n)
          k2 = f(q_n + Δt/2 · k1)
          k3 = f(q_n + Δt/2 · k2)
          k4 = f(q_n + Δt · k3)
          q_{n+1} = q_n + (Δt/6)(k1 + 2k2 + 2k3 + k4)

        优点：四阶精度，稳定性好，适合长时间积分
        缺点：每步需要 4 次力评估（计算量较大）

        参数:
            x0, y0:     初始位置 [m]
            vx0, vy0:   初始速度 [m/s]
            n_steps:    时间步数
        返回:
            trajectory: np.ndarray (n_steps+1, 5)，各列为 [t, x, y, vx, vy]
        """
        if n_steps is None:
            n_steps = params.n_steps

        dt = self.dt
        q  = np.array([x0, y0, vx0, vy0])

        trajectory = np.zeros((n_steps + 1, 5))
        trajectory[0] = [0.0, *q]

        print(f"  开始 RK4 积分: {n_steps} 步, dt={dt:.1e}s...")

        for step in range(n_steps):
            # RK4 四个阶段
            k1 = self._state_derivative(q)                      # 在 t_n 处评估
            k2 = self._state_derivative(q + 0.5 * dt * k1)     # 在 t_n + Δt/2 处
            k3 = self._state_derivative(q + 0.5 * dt * k2)     # 在 t_n + Δt/2 处
            k4 = self._state_derivative(q + dt * k3)            # 在 t_n + Δt 处

            # 加权平均更新
            q = q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t = (step + 1) * dt

            trajectory[step + 1] = [t, *q]

            if (step + 1) % max(1, n_steps // 10) == 0:
                print(f"    步 {step+1}/{n_steps}: "
                      f"x={q[0]*1e6:.2f}μm, y={q[1]*1e6:.2f}μm, "
                      f"|v|={np.sqrt(q[2]**2+q[3]**2)*1e6:.2f}μm/s")

        return trajectory


# ═══════════════════════════════════════════════════════════════════
# 辅助函数：轨迹统计
# ═══════════════════════════════════════════════════════════════════
def trajectory_summary(trajectory):
    """
    打印轨迹统计信息。

    参数:
        trajectory: np.ndarray (N, 5)，[t, x, y, vx, vy]
    """
    t  = trajectory[:, 0]                             # 时间
    x  = trajectory[:, 1]                             # x 位置
    y  = trajectory[:, 2]                             # y 位置
    vx = trajectory[:, 3]                             # x 速度
    vy = trajectory[:, 4]                             # y 速度

    # 总位移
    dx_total = x[-1] - x[0]
    dy_total = y[-1] - y[0]
    disp     = np.sqrt(dx_total**2 + dy_total**2)

    # 最大速度
    speed = np.sqrt(vx**2 + vy**2)
    v_max = speed.max()

    print(f"\n轨迹统计:")
    print(f"  总时间:     {t[-1]:.4f} s")
    print(f"  总位移:     {disp*1e6:.4f} μm")
    print(f"  最终位置:   ({x[-1]*1e6:.4f}, {y[-1]*1e6:.4f}) μm")
    print(f"  最大速度:   {v_max*1e6:.4f} μm/s")
    print(f"  x 方向位移: {dx_total*1e6:.4f} μm")
    print(f"  y 方向位移: {dy_total*1e6:.4f} μm")


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[dynamics v5] 模块加载成功")
    print(f"  固体质量:   {params.m_solid:.4e} kg")
    print(f"  阻力系数:   {params.drag_coeff:.4e} N·s/m")
    print(f"  时间步长:   {params.dt_dynamics:.1e} s")
    print(f"  总仿真时间: {params.t_total} s")
    print(f"  总步数:     {params.n_steps}")
    print("[dynamics v5] 自检通过 ✅")
