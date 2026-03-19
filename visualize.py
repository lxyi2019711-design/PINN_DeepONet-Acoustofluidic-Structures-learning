"""
visualize.py — 可视化模块（v5 声-流-固耦合完整版）
================================================================
【v5 升级说明】
  1. 新增 plot_pressure_field(): 压力场等值线图
  2. 新增 plot_velocity_vectors(): 速度矢量场（箭头图）
  3. 新增 plot_trajectory(): 微结构运动轨迹图
  4. 新增 plot_force_history(): 受力随时间变化图
  5. 保留原有损失曲线、流线图、误差图等功能
  6. 输出扩展为 (u, v, p) 对应的可视化

输出图片列表（每个工况）：
  velocity_pred_{f}MHz.png     — 预测速度大小场
  velocity_true_{f}MHz.png     — 真实速度大小场
  velocity_error_{f}MHz.png    — 绝对误差
  pressure_field_{f}MHz.png    — 预测压力场
  velocity_vectors_{f}MHz.png  — 速度矢量场
  streamline_{f}MHz.png        — 流线图
  trajectory.png               — 微结构运动轨迹
  loss_curves.png              — 训练损失曲线
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import os                                          # 文件操作
import numpy as np                                 # 数值计算
import matplotlib                                  # 绑定后端
matplotlib.use('Agg')                              # 非交互后端（服务器环境）
import matplotlib.pyplot as plt                    # 绑定绘图
import matplotlib.tri as tri                       # 三角剖分插值

from config import params                          # 物理参数
from geometry import in_triangle                   # 几何工具

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════
def _to_numpy(arr):
    """将 Tensor 或 ndarray 统一转为 numpy。"""
    if hasattr(arr, 'numpy'):
        return arr.cpu().numpy()
    return np.asarray(arr)


def _scatter_to_grid(x_norm, y_norm, values, nx=200, ny=120):
    """
    散点数据插值到规则网格（Delaunay 三角剖分线性插值）。
    三角形固体区域置 NaN（遮罩）。
    """
    xi = np.linspace(0, 1, nx)
    yi = np.linspace(0, 1, ny)
    xi, yi = np.meshgrid(xi, yi)

    triang = tri.Triangulation(x_norm, y_norm)
    zi     = tri.LinearTriInterpolator(triang, values)(xi, yi)

    # 固体遮罩
    x_phys = xi * params.Lx + params.Lx_min
    y_phys = yi * params.Ly
    zi[in_triangle(x_phys.ravel(), y_phys.ravel()).reshape(xi.shape)] = np.nan

    return xi, yi, np.ma.array(zi, mask=np.isnan(zi))


def _draw_triangle_patch(ax, color='gray', alpha=0.85):
    """在归一化坐标轴上绘制三角形固体结构。"""
    cx = (params.tri_cx - params.Lx_min) / params.Lx
    by =  params.tri_base_y / params.Ly
    b  =  params.tri_base   / params.Lx
    h  =  params.tri_height / params.Ly
    patch = plt.Polygon(
        [[cx - b/2, by], [cx + b/2, by], [cx, by + h]],
        closed=True, facecolor=color, edgecolor='k',
        linewidth=0.8, alpha=alpha, zorder=5
    )
    ax.add_patch(patch)


# ═══════════════════════════════════════════════════════════════════
# 损失曲线
# ═══════════════════════════════════════════════════════════════════
def plot_loss_curves(history: dict, save_path: str = 'loss_curves.png'):
    """
    绘制训练损失曲线（总损失 + 各物理约束分项）。
    """
    has_mom = any(v > 0 for v in history.get('momentum', [0]))
    n_cols  = 3 if has_mom else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))

    # 总损失
    ax = axes[0]
    ax.semilogy(history['train_loss'], label='Train Loss', color='steelblue')
    ax.semilogy(history['val_loss'],   label='Val Loss',   color='tomato', linestyle='--')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    # PDE 分项
    ax = axes[1]
    ax.semilogy(history['fluid_data'], label='Data Loss',       color='steelblue')
    ax.semilogy(history['continuity'], label='Continuity',      color='orange')
    if has_mom:
        ax.semilogy(history['momentum'], label='Momentum(NS)', color='green')
    ax.set_title('PDE Loss Components')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    # 自适应权重
    if has_mom and n_cols == 3:
        ax = axes[2]
        ax.plot(history.get('w_data', []), label='w_data', color='steelblue')
        ax.plot(history.get('w_pde',  []), label='w_pde',  color='orange')
        ax.set_title('Adaptive Weights')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 速度场等值线图
# ═══════════════════════════════════════════════════════════════════
def plot_velocity_fields(X_test_raw, Y_pred_real, Y_true_real,
                         freq_mhz: float, out_dir: str = '.'):
    """
    绘制预测/真实速度场和误差场。
    v5: 速度大小 = sqrt(u²+v²)
    """
    os.makedirs(out_dir, exist_ok=True)
    X_np = _to_numpy(X_test_raw)
    x_n, y_n = X_np[:, 0], X_np[:, 1]
    freq_str = f"{freq_mhz:.2f}MHz"

    # 计算速度大小
    vel_pred = np.sqrt(Y_pred_real[:, 0]**2 + Y_pred_real[:, 1]**2)
    vel_true = np.sqrt(Y_true_real[:, 0]**2 + Y_true_real[:, 1]**2)
    vel_error = vel_pred - vel_true

    for values, cmap, title, fname, sym in [
        (vel_pred,  'viridis', 'Predicted |V|', f'velocity_pred_{freq_str}.png',  False),
        (vel_true,  'viridis', 'Ground Truth |V|', f'velocity_true_{freq_str}.png',  False),
        (vel_error, 'RdBu_r',  'Error |V|_pred - |V|_true', f'velocity_error_{freq_str}.png', True),
    ]:
        xi, yi, zi = _scatter_to_grid(x_n, y_n, values)
        fig, ax = plt.subplots(figsize=(9, 5))
        vmin, vmax = (None, None)
        if sym:
            vabs = max(abs(np.nanmin(zi)), abs(np.nanmax(zi)))
            vmin, vmax = -vabs, vabs
        cf = ax.contourf(xi, yi, zi, levels=60, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(cf, ax=ax, label='m/s')
        _draw_triangle_patch(ax)
        ax.set_xlabel('x / Lx'); ax.set_ylabel('y / Ly')
        ax.set_title(f'{title}  [{freq_str}]')
        ax.set_aspect('equal')
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 压力场等值线图（v5 新增）
# ═══════════════════════════════════════════════════════════════════
def plot_pressure_field(X_test_raw, Y_pred_real, Y_true_real,
                        freq_mhz: float, out_dir: str = '.'):
    """
    绘制预测压力场分布。
    """
    os.makedirs(out_dir, exist_ok=True)
    X_np = _to_numpy(X_test_raw)
    x_n, y_n = X_np[:, 0], X_np[:, 1]
    freq_str = f"{freq_mhz:.2f}MHz"

    p_pred = Y_pred_real[:, 2]                      # 预测压力
    p_true = Y_true_real[:, 2]                      # 真实压力

    for values, cmap, title, fname in [
        (p_pred, 'coolwarm', 'Predicted Pressure', f'pressure_pred_{freq_str}.png'),
        (p_true, 'coolwarm', 'Ground Truth Pressure', f'pressure_true_{freq_str}.png'),
    ]:
        xi, yi, zi = _scatter_to_grid(x_n, y_n, values)
        fig, ax = plt.subplots(figsize=(9, 5))
        cf = ax.contourf(xi, yi, zi, levels=60, cmap=cmap)
        plt.colorbar(cf, ax=ax, label='Pa')
        _draw_triangle_patch(ax)
        ax.set_xlabel('x / Lx'); ax.set_ylabel('y / Ly')
        ax.set_title(f'{title}  [{freq_str}]')
        ax.set_aspect('equal')
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 速度矢量场（v5 新增）
# ═══════════════════════════════════════════════════════════════════
def plot_velocity_vectors(X_test_raw, Y_pred_real,
                          freq_mhz: float, out_dir: str = '.', stride: int = 6):
    """
    绘制速度矢量场（箭头图）。
    背景为速度大小等值线，箭头表示速度方向和大小。
    """
    os.makedirs(out_dir, exist_ok=True)
    X_np = _to_numpy(X_test_raw)
    x_n, y_n = X_np[:, 0], X_np[:, 1]
    freq_str = f"{freq_mhz:.2f}MHz"

    u_pred = Y_pred_real[:, 0]
    v_pred = Y_pred_real[:, 1]
    speed  = np.sqrt(u_pred**2 + v_pred**2)

    # 背景速度大小
    xi, yi, zi_speed = _scatter_to_grid(x_n, y_n, speed)

    # 矢量场网格
    nx_q, ny_q = 50, 30
    _, _, zi_u = _scatter_to_grid(x_n, y_n, u_pred, nx_q, ny_q)
    _, _, zi_v = _scatter_to_grid(x_n, y_n, v_pred, nx_q, ny_q)

    xi_q = np.linspace(0, 1, nx_q)
    yi_q = np.linspace(0, 1, ny_q)
    xq, yq = np.meshgrid(xi_q[::stride], yi_q[::stride])
    uq = np.ma.filled(zi_u, 0)[::stride, ::stride]
    vq = np.ma.filled(zi_v, 0)[::stride, ::stride]

    fig, ax = plt.subplots(figsize=(9, 5))
    cf = ax.contourf(xi, yi, zi_speed, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(cf, ax=ax, label='|V| [m/s]')
    ax.quiver(xq, yq, uq, vq, color='white', alpha=0.8, width=0.003,
              scale=None, scale_units='xy')
    _draw_triangle_patch(ax)
    ax.set_xlabel('x / Lx'); ax.set_ylabel('y / Ly')
    ax.set_title(f'Velocity Vectors  [{freq_str}]')
    ax.set_aspect('equal')
    plt.tight_layout()
    fname = os.path.join(out_dir, f'velocity_vectors_{freq_str}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
# 流线图
# ═══════════════════════════════════════════════════════════════════
def plot_streamlines(X_test_raw, Y_pred_real,
                     freq_mhz: float, out_dir: str = '.'):
    """根据预测 (u, v) 绘制流线图。"""
    os.makedirs(out_dir, exist_ok=True)
    X_np = _to_numpy(X_test_raw)
    x_n, y_n = X_np[:, 0], X_np[:, 1]
    freq_str = f"{freq_mhz:.2f}MHz"

    nx, ny = 150, 90
    xi, yi, ui = _scatter_to_grid(x_n, y_n, Y_pred_real[:, 0], nx, ny)
    _, _, vi   = _scatter_to_grid(x_n, y_n, Y_pred_real[:, 1], nx, ny)

    ui_f = np.ma.filled(ui, 0.0)
    vi_f = np.ma.filled(vi, 0.0)
    speed = np.sqrt(ui_f**2 + vi_f**2)

    fig, ax = plt.subplots(figsize=(9, 5))
    strm = ax.streamplot(xi[0, :], yi[:, 0], ui_f, vi_f,
                         color=speed, cmap='plasma',
                         linewidth=0.8, density=1.5, arrowsize=0.8)
    plt.colorbar(strm.lines, ax=ax, label='|V| [m/s]')
    _draw_triangle_patch(ax, color='dimgray')
    ax.set_xlabel('x / Lx'); ax.set_ylabel('y / Ly')
    ax.set_title(f'Streamlines (Predicted)  [{freq_str}]')
    ax.set_aspect('equal')
    plt.tight_layout()
    fname = os.path.join(out_dir, f'streamline_{freq_str}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
# 微结构运动轨迹图（v5 新增）
# ═══════════════════════════════════════════════════════════════════
def plot_trajectory(trajectory, save_path='trajectory.png'):
    """
    绘制微结构运动轨迹。

    四子图：
      1. x-y 平面轨迹
      2. x(t) 位移随时间
      3. y(t) 位移随时间
      4. |v(t)| 速度大小随时间

    参数:
        trajectory: np.ndarray (N, 5)，各列 [t, x, y, vx, vy]
        save_path:  保存路径
    """
    t  = trajectory[:, 0] * 1e3                     # 时间 → ms
    x  = trajectory[:, 1] * 1e6                     # 位置 → μm
    y  = trajectory[:, 2] * 1e6
    vx = trajectory[:, 3] * 1e6                     # 速度 → μm/s
    vy = trajectory[:, 4] * 1e6
    speed = np.sqrt(vx**2 + vy**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. 轨迹图
    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=t, cmap='viridis', s=2, alpha=0.8)
    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=8, label='End')
    plt.colorbar(sc, ax=ax, label='Time [ms]')
    ax.set_xlabel('x [um]'); ax.set_ylabel('y [um]')
    ax.set_title('Microstructure Trajectory')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. x(t)
    ax = axes[0, 1]
    ax.plot(t, x, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('x [um]')
    ax.set_title('x Displacement')
    ax.grid(True, alpha=0.3)

    # 3. y(t)
    ax = axes[1, 0]
    ax.plot(t, y, color='tomato', linewidth=1.5)
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('y [um]')
    ax.set_title('y Displacement')
    ax.grid(True, alpha=0.3)

    # 4. |v(t)|
    ax = axes[1, 1]
    ax.plot(t, speed, color='green', linewidth=1.5)
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('|v| [um/s]')
    ax.set_title('Speed')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Microstructure Dynamics in Acoustic Streaming', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 综合摘要图
# ═══════════════════════════════════════════════════════════════════
def plot_summary(X_test_raw, Y_pred_real, Y_true_real, history,
                 freq_mhz: float = 0.55, save_path: str = 'summary.png'):
    """综合摘要图（2×3）：损失 + u/v/p 散点对比。"""
    X_np = _to_numpy(X_test_raw)
    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    # 损失曲线
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.semilogy(history['train_loss'], label='Train', color='steelblue')
    ax0.semilogy(history['val_loss'],   label='Val', color='tomato', ls='--')
    ax0.set_title('Loss'); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

    # u, v, p 散点对比
    field_labels = ['u [m/s]', 'v [m/s]', 'p [Pa]']
    colors = ['steelblue', 'darkorange', 'green']
    positions = [(0,1),(0,2),(1,0)]
    for idx, (r, c) in enumerate(positions):
        ax = fig.add_subplot(gs[r, c])
        pred, true = Y_pred_real[:, idx], Y_true_real[:, idx]
        ax.scatter(true, pred, s=2, alpha=0.4, color=colors[idx])
        lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lim, lim, 'k--', linewidth=0.8)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        ax.set_title(field_labels[idx])

    # 误差信息
    ax_info = fig.add_subplot(gs[1, 1:])
    ax_info.axis('off')
    info_text = f'Frequency: {freq_mhz:.2f} MHz\n'
    for idx, name in enumerate(field_labels):
        err = (np.linalg.norm(Y_pred_real[:, idx] - Y_true_real[:, idx]) /
               (np.linalg.norm(Y_true_real[:, idx]) + 1e-10) * 100)
        info_text += f'{name} Rel L2 Error: {err:.2f}%\n'
    ax_info.text(0.3, 0.5, info_text, ha='left', va='center', fontsize=12,
                 transform=ax_info.transAxes, family='monospace')

    plt.suptitle(f'FP-PiDON Summary  [{freq_mhz:.2f}MHz]', fontsize=13)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 批量输出
# ═══════════════════════════════════════════════════════════════════
def save_all_figures(X_test_raw, Y_pred_real, Y_true_real,
                     freq_mhz: float, out_dir: str = 'results'):
    """
    输出单个工况的全部可视化图片。
    """
    print(f"\n[visualize] Plotting for {freq_mhz:.2f} MHz...")
    plot_velocity_fields(X_test_raw, Y_pred_real, Y_true_real, freq_mhz, out_dir)
    plot_pressure_field( X_test_raw, Y_pred_real, Y_true_real, freq_mhz, out_dir)
    plot_velocity_vectors(X_test_raw, Y_pred_real, freq_mhz, out_dir)
    plot_streamlines(    X_test_raw, Y_pred_real, freq_mhz, out_dir)
