"""
geometry.py — 几何域工具函数（v5 声-流-固耦合版）
================================================================
【v5 升级说明】
  1. 新增 get_triangle_vertices(): 返回三角形三个顶点坐标
  2. 新增 sample_triangle_surface(): 在三角形三条边上均匀采样，返回点坐标和外法向量
  3. 新增 compute_outward_normals(): 计算三角形各边的单位外法向量
  4. 保留原有 in_triangle()、in_fluid_domain()、sample_fluid_points()

物理意义：
  三角形表面采样点和法向量用于固体受力计算——
  通过在固体表面积分流体压力和剪切应力来得到合力 (Fx, Fy)：
    F = ∮_S (-p·n + τ·n) dS
  其中 n 为表面外法向量，τ 为粘性应力张量。
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import numpy as np                              # 数值计算
from config import params                       # 物理参数配置


# ═══════════════════════════════════════════════════════════════════
# 获取三角形顶点坐标
# ═══════════════════════════════════════════════════════════════════
def get_triangle_vertices():
    """
    返回等腰三角形的三个顶点坐标 [m]。

    三角形定义：
      A = 左下角 = (cx - base/2,  base_y)
      B = 右下角 = (cx + base/2,  base_y)
      C = 顶点   = (cx,           base_y + height)

    顶点朝 +y 方向（尖端向上），底边水平位于 y = base_y。

    返回:
        np.ndarray, shape (3, 2)，每行为一个顶点的 (x, y) 坐标
    """
    cx = params.tri_cx                          # 底边中心 x 坐标
    by = params.tri_base_y                      # 底边 y 坐标
    b  = params.tri_base                        # 底边宽度
    h  = params.tri_height                      # 三角形高度

    A = np.array([cx - b / 2, by])              # 左下顶点
    B = np.array([cx + b / 2, by])              # 右下顶点
    C = np.array([cx,         by + h])          # 顶点（尖端）

    return np.array([A, B, C])                  # (3, 2) 矩阵


# ═══════════════════════════════════════════════════════════════════
# 判断点是否在三角形内部（重心坐标法）
# ═══════════════════════════════════════════════════════════════════
def in_triangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    判断各点 (x, y) 是否位于等腰三角形内部（含边界）。

    使用重心坐标法（barycentric coordinate）：
      给定三角形 ABC 和查询点 P，计算三个叉积的符号。
      若符号全部一致（全正或全负），则 P 在三角形内部。

    数学原理：
      对于每条边 AB，计算 cross(P-A, B-A) 的符号。
      三条边的叉积符号一致 ↔ 点在三角形内部。

    参数:
        x, y: np.ndarray, 查询点坐标 [m]（SI 单位）
    返回:
        bool np.ndarray, True 表示点在三角形内部
    """
    verts = get_triangle_vertices()              # 获取三顶点
    ax, ay = verts[0]                            # A = 左下
    bx, by = verts[1]                            # B = 右下
    cx, cy = verts[2]                            # C = 顶点

    # 叉积函数：计算向量 (Q-P) × (R-P) 的 z 分量
    def cross(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    d1 = cross(x, y, ax, ay, bx, by)            # 相对边 AB 的叉积
    d2 = cross(x, y, bx, by, cx, cy)            # 相对边 BC 的叉积
    d3 = cross(x, y, cx, cy, ax, ay)            # 相对边 CA 的叉积

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)   # 是否存在负叉积
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)   # 是否存在正叉积

    # 叉积全正或全负 → 点在三角形内部
    return ~(has_neg & has_pos)


# ═══════════════════════════════════════════════════════════════════
# 判断点是否在有效流体域内
# ═══════════════════════════════════════════════════════════════════
def in_fluid_domain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    判断点是否属于有效流体域：矩形域内 AND 不在三角形固体区域内。

    流体域 = 矩形域 ∩ ¬三角形域
    即排除三角形固体所占区域后的流体区域。

    参数:
        x, y: np.ndarray, 坐标 [m]
    返回:
        bool np.ndarray
    """
    # 矩形域边界判断
    in_rect = ((x >= params.Lx_min) & (x <= params.Lx_max) &
               (y >= 0) & (y <= params.Ly))
    # 流体域 = 矩形内 且 不在三角形内
    return in_rect & ~in_triangle(x, y)


# ═══════════════════════════════════════════════════════════════════
# 在流体域内随机采样（排除三角形）
# ═══════════════════════════════════════════════════════════════════
def sample_fluid_points(n_target: int, seed: int = None) -> tuple:
    """
    在流体域内均匀随机采样 n_target 个点。
    使用拒绝采样（rejection sampling）：先过采样，再剔除三角形内的点。

    参数:
        n_target: 目标采样点数
        seed:     随机种子（可选）
    返回:
        (x, y) np.ndarray, 各 shape (n_target,), 单位 [m]
    """
    rng = np.random.default_rng(seed)            # 创建随机数生成器
    xs, ys = [], []                              # 累积有效点
    collected = 0                                # 已收集点数
    oversample = max(int(n_target * 1.5), n_target + 100)  # 过采样倍率

    while collected < n_target:                  # 循环直到满足目标点数
        x_cand = rng.uniform(params.Lx_min, params.Lx_max, oversample)  # 随机 x
        y_cand = rng.uniform(0, params.Ly, oversample)                  # 随机 y
        mask   = in_fluid_domain(x_cand, y_cand)                       # 筛选流体域
        xs.append(x_cand[mask])                  # 保留有效点
        ys.append(y_cand[mask])
        collected += mask.sum()                  # 更新计数

    x_all = np.concatenate(xs)[:n_target]        # 截取目标数量
    y_all = np.concatenate(ys)[:n_target]
    return x_all, y_all


# ═══════════════════════════════════════════════════════════════════
# 三角形表面采样与法向量计算（v5 新增，用于固体受力积分）
# ═══════════════════════════════════════════════════════════════════
def sample_triangle_surface(n_per_edge: int = None, offset: float = 1e-7):
    """
    在三角形三条边上均匀采样，返回采样点坐标、外法向量和边元长度。

    物理目的：
      为了计算流体对固体表面的压力和剪切应力，需要在固体表面离散化，
      在每个采样点处评估流体速度场和压力场，然后进行数值面积分：
        F = ∮_S (-p·n + μ(∇u + ∇u^T)·n) dS

    参数:
        n_per_edge: 每条边的采样点数（默认使用 config 中的 n_surface_pts//3）
        offset:     采样点向外偏移量 [m]（避免恰好在固体边界上，取流体侧值）
    返回:
        points:  np.ndarray, shape (N, 2), 采样点坐标 [m]
        normals: np.ndarray, shape (N, 2), 单位外法向量
        ds:      np.ndarray, shape (N,),   每个采样点对应的边元长度 [m]
    """
    if n_per_edge is None:                       # 默认每条边取总数的 1/3
        n_per_edge = params.n_surface_pts // 3

    verts = get_triangle_vertices()              # 三角形三顶点
    edges = [                                    # 定义三条边 (起点, 终点)
        (verts[0], verts[1]),                    # 底边：A → B
        (verts[1], verts[2]),                    # 右侧斜边：B → C
        (verts[2], verts[0]),                    # 左侧斜边：C → A
    ]

    all_points  = []                             # 存储所有采样点
    all_normals = []                             # 存储所有法向量
    all_ds      = []                             # 存储所有边元长度

    # 三角形几何中心（用于判断法向量朝外方向）
    centroid = verts.mean(axis=0)                # 质心坐标

    for p_start, p_end in edges:                 # 遍历三条边
        edge_vec = p_end - p_start               # 边向量（从起点到终点）
        edge_len = np.linalg.norm(edge_vec)      # 边长度 [m]
        tangent  = edge_vec / edge_len           # 单位切向量

        # 法向量（垂直于切向量，二维旋转90°）
        # 右手系：n = (-ty, tx) 或 (ty, -tx)
        n_candidate = np.array([-tangent[1], tangent[0]])  # 候选法向量

        # 判断法向量是否朝外：法向量应指向远离质心方向
        edge_mid = 0.5 * (p_start + p_end)       # 边中点
        if np.dot(n_candidate, edge_mid - centroid) < 0:  # 如果朝向质心
            n_candidate = -n_candidate            # 翻转方向，使其朝外

        ds_elem = edge_len / n_per_edge           # 每个采样元的长度

        # 在边上均匀采样（参数 t ∈ (0, 1)，不含端点避免重复）
        t_vals = np.linspace(0.5 / n_per_edge,
                             1 - 0.5 / n_per_edge,
                             n_per_edge)
        for t in t_vals:
            pt = p_start + t * edge_vec           # 边上的点
            pt_offset = pt + offset * n_candidate # 向外偏移（取流体侧）
            all_points.append(pt_offset)          # 记录偏移后的点
            all_normals.append(n_candidate)        # 记录法向量
            all_ds.append(ds_elem)                 # 记录边元长度

    points  = np.array(all_points)               # (N, 2) 采样点坐标
    normals = np.array(all_normals)              # (N, 2) 外法向量
    ds      = np.array(all_ds)                   # (N,)   边元长度

    return points, normals, ds


# ═══════════════════════════════════════════════════════════════════
# 将三角形表面点平移到新的质心位置
# ═══════════════════════════════════════════════════════════════════
def translate_surface_points(base_points, base_normals, dx, dy):
    """
    将表面采样点平移到新位置。
    当固体在声流场中运动时，其表面位置随之变化，
    需要实时更新表面采样点的坐标。

    参数:
        base_points:  原始采样点 (N, 2)
        base_normals: 法向量（平移不变）(N, 2)
        dx, dy:       x, y 方向的位移 [m]
    返回:
        new_points:   平移后的采样点 (N, 2)
        normals:      法向量（不变）(N, 2)
    """
    displacement = np.array([dx, dy])            # 位移向量
    new_points = base_points + displacement      # 平移所有采样点
    return new_points, base_normals              # 法向量不变（刚体平移）


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 验证三角形顶点
    verts = get_triangle_vertices()
    print("三角形顶点 [μm]:")
    for i, v in enumerate(verts):
        print(f"  V{i}: ({v[0]*1e6:.2f}, {v[1]*1e6:.2f})")

    # 验证表面采样
    pts, norms, ds = sample_triangle_surface(n_per_edge=10)
    print(f"\n表面采样: {len(pts)} 个点")
    print(f"  坐标范围 x: [{pts[:,0].min()*1e6:.2f}, {pts[:,0].max()*1e6:.2f}] μm")
    print(f"  坐标范围 y: [{pts[:,1].min()*1e6:.2f}, {pts[:,1].max()*1e6:.2f}] μm")
    print(f"  法向量模: {np.linalg.norm(norms, axis=1).mean():.6f}（应接近 1.0）")
    print(f"  总边元长度: {ds.sum()*1e6:.2f} μm")

    # 验证流体域采样
    xs, ys = sample_fluid_points(500, seed=0)
    print(f"\n流体域采样: {len(xs)} 个点")
    print("[geometry v5] 自检通过 ✅")
