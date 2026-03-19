"""
network_viz.py — 网络结构可视化模块（v5 新增模块）
================================================================
本模块自动绘制 FP-PiDON (DeepONet) 的网络架构示意图。

提供两种可视化方式：
  1. 使用 matplotlib 手动绘制结构示意图（无外部依赖）
  2. 使用 graphviz（如果已安装）绘制精细的有向图

网络结构：
  ┌─────────────────────────────────┐
  │         FP-PiDON Model          │
  │                                 │
  │  ┌──────────┐  ┌──────────────┐ │
  │  │ BranchNet│  │  TrunkNet    │ │
  │  │          │  │              │ │
  │  │ freq(1)  │  │ Fourier Enc │ │
  │  │  ↓       │  │  (x,y) →    │ │
  │  │ MLP×4    │  │  [cos,sin]  │ │
  │  │  ↓       │  │    ↓        │ │
  │  │ (3×128)  │  │ ResBlock×6  │ │
  │  └────┬─────┘  │    ↓        │ │
  │       │        │  (128)      │ │
  │       │        └──────┬──────┘ │
  │       │               │        │
  │       └───── ⊗ ───────┘        │  ← einsum dot product
  │               │                │
  │          + bias (3)            │
  │               ↓                │
  │          [u, v, p]             │
  └─────────────────────────────────┘
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import os                                          # 文件操作
import numpy as np                                 # 数值计算
import matplotlib                                  # matplotlib
matplotlib.use('Agg')                              # 非交互后端
import matplotlib.pyplot as plt                    # 绑定绘图
import matplotlib.patches as mpatches              # 矩形、圆形等图形


# ═══════════════════════════════════════════════════════════════════
# 方式一：Matplotlib 手绘结构示意图
# ═══════════════════════════════════════════════════════════════════
def plot_architecture_matplotlib(save_path='network_architecture.png'):
    """
    使用 matplotlib 绘制 FP-PiDON 网络架构示意图。
    无需安装 graphviz，纯 Python 实现。

    图示结构：
      左侧：BranchNet（频率输入 → MLP → 系数矩阵）
      右侧：TrunkNet（空间坐标 → Fourier编码 → 残差网络 → 基函数）
      中央：内积运算 + 偏置 → 输出 (u, v, p)
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-2, 12)
    ax.axis('off')

    # 颜色方案
    c_branch = '#4A90D9'                            # Branch 蓝色系
    c_trunk  = '#E67E22'                            # Trunk 橙色系
    c_output = '#2ECC71'                            # 输出 绿色
    c_input  = '#9B59B6'                            # 输入 紫色
    c_op     = '#E74C3C'                            # 运算 红色

    def draw_box(x, y, w, h, text, color, fontsize=9):
        """绘制圆角矩形框。"""
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')

    def draw_arrow(x1, y1, x2, y2, color='black'):
        """绘制箭头。"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, connectionstyle='arc3,rad=0'))

    # ── 标题 ──
    ax.text(8, 11.5, 'FP-PiDON: Frequency-Parameterized Physics-Informed DeepONet',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(8, 10.8, r'$G(f)(x,y) = \sum_j b_{k,j}(f) \cdot t_j(x,y) + \mathrm{bias}_k$',
            ha='center', va='center', fontsize=12, style='italic')

    # ═══ 左侧：Branch Net ═══
    # 输入
    draw_box(0, 8.5, 3, 0.8, 'freq (1D)', c_input)
    ax.text(1.5, 9.7, 'Branch Net', ha='center', fontsize=11, fontweight='bold', color=c_branch)

    # MLP 层
    draw_box(0, 7, 3, 0.8, 'Linear(1→64) + Tanh', c_branch)
    draw_arrow(1.5, 8.5, 1.5, 7.8)

    draw_box(0, 5.5, 3, 0.8, 'Linear(64→64) + Tanh ×3', c_branch)
    draw_arrow(1.5, 7, 1.5, 6.3)

    draw_box(0, 4, 3, 0.8, 'Linear(64→384)', c_branch)
    draw_arrow(1.5, 5.5, 1.5, 4.8)

    draw_box(0, 2.5, 3, 0.8, 'Reshape → (3, 128)', c_branch)
    draw_arrow(1.5, 4, 1.5, 3.3)

    # ═══ 右侧：Trunk Net ═══
    draw_box(10, 8.5, 4, 0.8, '(x, y) (2D)', c_input)
    ax.text(12, 9.7, 'Trunk Net', ha='center', fontsize=11, fontweight='bold', color=c_trunk)

    draw_box(10, 7, 4, 0.8, 'Fourier Encode → (128D)', c_trunk)
    draw_arrow(12, 8.5, 12, 7.8)
    ax.text(14.5, 7.4, r'$\gamma(\mathbf{x})=[\cos(B\mathbf{x}), \sin(B\mathbf{x})]$',
            fontsize=8, style='italic')

    draw_box(10, 5.5, 4, 0.8, 'Linear(128→256) + Tanh', c_trunk)
    draw_arrow(12, 7, 12, 6.3)

    draw_box(10, 4, 4, 0.8, 'ResidualBlock ×6', c_trunk)
    draw_arrow(12, 5.5, 12, 4.8)
    ax.text(14.5, 4.4, 'Skip Connections', fontsize=8, style='italic')

    draw_box(10, 2.5, 4, 0.8, 'Linear(256→128) + Tanh', c_trunk)
    draw_arrow(12, 4, 12, 3.3)

    # ═══ 中央：内积运算 ═══
    # 绘制圆形运算符号
    circle = plt.Circle((6.5, 1.5), 0.4, color=c_op, alpha=0.9, zorder=10)
    ax.add_patch(circle)
    ax.text(6.5, 1.5, r'$\otimes$', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white', zorder=11)
    ax.text(6.5, 0.9, 'einsum(nkp, np→nk)', ha='center', fontsize=8)

    # Branch → 内积
    draw_arrow(1.5, 2.5, 6.1, 1.7, c_branch)
    # Trunk → 内积
    draw_arrow(12, 2.5, 6.9, 1.7, c_trunk)

    # ═══ 输出 ═══
    draw_box(5, -0.5, 3, 0.8, '+ bias (3D)', c_op)
    draw_arrow(6.5, 1.1, 6.5, 0.3)

    draw_box(5, -1.8, 3, 0.8, 'Output: [u, v, p]', c_output)
    draw_arrow(6.5, -0.5, 6.5, -1)

    # ═══ PINN 物理约束标注 ═══
    # 右下角标注物理损失
    pinn_x, pinn_y = 11, -0.5
    ax.text(pinn_x, pinn_y + 0.5, 'PINN Physics Constraints:', fontsize=10,
            fontweight='bold', color='#333')
    constraints = [
        r'$\nabla \cdot \mathbf{u} = 0$ (Continuity)',
        r'$-\nabla p + \mu \nabla^2 \mathbf{u} = 0$ (Stokes)',
        r'$\mathbf{u}|_{wall} = 0$ (No-slip BC)',
    ]
    for i, c in enumerate(constraints):
        ax.text(pinn_x, pinn_y - 0.4 * i, c, fontsize=9, color='#555')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Network architecture saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 方式二：Graphviz 有向图（如果已安装）
# ═══════════════════════════════════════════════════════════════════
def plot_architecture_graphviz(save_path='network_graph'):
    """
    使用 graphviz 绘制 DeepONet 网络结构有向图。
    需要安装 graphviz：pip install graphviz

    生成 .png 和 .pdf 两种格式。
    """
    try:
        from graphviz import Digraph                # 导入 graphviz
    except ImportError:
        print("  [Warning] graphviz not installed. Skipping graph visualization.")
        print("  Install with: pip install graphviz")
        return

    dot = Digraph('FP-PiDON', format='png')
    dot.attr(rankdir='TB', size='12,15')
    dot.attr('node', shape='box', style='rounded,filled', fontsize='10')

    # 子图：Branch Net
    with dot.subgraph(name='cluster_branch') as c:
        c.attr(label='Branch Net (Frequency Encoder)',
               style='dashed', color='blue', fontcolor='blue')
        c.node('freq_in', 'freq (1D)', fillcolor='#D5E8D4')
        c.node('b_mlp1', 'Linear(1→64)+Tanh', fillcolor='#DAE8FC')
        c.node('b_mlp2', 'Linear(64→64)+Tanh ×3', fillcolor='#DAE8FC')
        c.node('b_out', 'Linear(64→384)\nReshape→(3,128)', fillcolor='#DAE8FC')
        c.edge('freq_in', 'b_mlp1')
        c.edge('b_mlp1', 'b_mlp2')
        c.edge('b_mlp2', 'b_out')

    # 子图：Trunk Net
    with dot.subgraph(name='cluster_trunk') as c:
        c.attr(label='Trunk Net (Spatial Encoder)',
               style='dashed', color='orange', fontcolor='orange')
        c.node('xy_in', '(x, y) (2D)', fillcolor='#D5E8D4')
        c.node('fourier', 'Fourier Encode\n[cos(Bx), sin(Bx)]→128D', fillcolor='#FFF2CC')
        c.node('t_proj', 'Linear(128→256)+Tanh', fillcolor='#FFE6CC')
        c.node('t_res', 'ResidualBlock ×6\n(Skip Connections)', fillcolor='#FFE6CC')
        c.node('t_out', 'Linear(256→128)+Tanh', fillcolor='#FFE6CC')
        c.edge('xy_in', 'fourier')
        c.edge('fourier', 't_proj')
        c.edge('t_proj', 't_res')
        c.edge('t_res', 't_out')

    # 内积和输出
    dot.node('dot', '⊗ einsum\nbranch[n,k,p]·trunk[n,p]→[n,k]',
             shape='circle', fillcolor='#F8CECC', width='1.5')
    dot.node('bias', '+ bias (3D)', fillcolor='#E1D5E7')
    dot.node('output', 'Output: [u, v, p]', fillcolor='#D5E8D4',
             shape='doubleoctagon')

    dot.edge('b_out', 'dot', label='(N,3,128)')
    dot.edge('t_out', 'dot', label='(N,128)')
    dot.edge('dot', 'bias', label='(N,3)')
    dot.edge('bias', 'output')

    # PINN 标注
    dot.node('pinn', 'PINN Constraints:\n∇·u=0 | -∇p+μ∇²u=0 | u|wall=0',
             shape='note', fillcolor='#FFFFCC', fontsize='9')
    dot.edge('output', 'pinn', style='dashed', label='Physics Loss')

    # 渲染
    try:
        dot.render(save_path, cleanup=True)
        print(f"  Graphviz architecture saved: {save_path}.png")
    except Exception as e:
        print(f"  [Warning] Graphviz render failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# 模型参数统计图
# ═══════════════════════════════════════════════════════════════════
def plot_parameter_distribution(model, save_path='param_dist.png'):
    """
    绘制模型各层参数的分布直方图。
    可用于检查初始化是否合理、训练后参数是否过大/过小。
    """
    named_params = [(n, p.detach().cpu().numpy().ravel())
                    for n, p in model.named_parameters() if p.requires_grad]

    n_params = len(named_params)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = np.array(axes).ravel()

    for i, (name, vals) in enumerate(named_params):
        ax = axes[i]
        ax.hist(vals, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Parameter Distributions', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════
def generate_all_architecture_plots(model=None, out_dir='results'):
    """
    生成所有网络结构相关图片。
    """
    os.makedirs(out_dir, exist_ok=True)

    # Matplotlib 结构示意图（始终可用）
    plot_architecture_matplotlib(
        save_path=os.path.join(out_dir, 'network_architecture.png'))

    # Graphviz 有向图（可选）
    plot_architecture_graphviz(
        save_path=os.path.join(out_dir, 'network_graph'))

    # 参数分布（如果提供了模型）
    if model is not None:
        plot_parameter_distribution(
            model, save_path=os.path.join(out_dir, 'param_distribution.png'))


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs('test_viz', exist_ok=True)
    plot_architecture_matplotlib('test_viz/arch_mpl.png')
    plot_architecture_graphviz('test_viz/arch_gv')
    print("[network_viz v5] self-test passed ✅")
