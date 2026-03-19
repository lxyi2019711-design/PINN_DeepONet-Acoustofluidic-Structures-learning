"""
model.py — DeepONet + PINN 网络架构（v5 声-流-固耦合版）
================================================================
【v5 升级说明】
  1. 网络输出从 (velocity, u, v) 改为 (u, v, p)：速度分量 + 压力场
  2. 保留 ResidualBlock + Fourier 特征编码架构
  3. 添加完整的算子学习原理注释和数学公式

═══════════════════════════════════════════════════════════════════
DeepONet 算子学习原理（Operator Learning）
═══════════════════════════════════════════════════════════════════

核心思想：学习从"函数空间到函数空间"的映射。

传统神经网络学习：x → y（有限维向量之间的映射）
DeepONet 学习：   G(u)(y) — 算子 G 将输入函数 u 映射到输出函数在 y 点处的值

在本项目中：
  输入函数空间：声场参数空间（由频率 f 参数化）
  输出函数空间：时均声流场 (u, v, p) 在空间 (x, y) 处的值
  算子映射：     G: f → {u(x,y), v(x,y), p(x,y)}

DeepONet 由两个子网络组成：
  Branch Net：编码输入函数（频率参数）→ 系数 {b_k}
  Trunk Net： 编码输出位置（空间坐标）→ 基函数 {t_k}
  最终输出：  G(f)(x,y) = Σ_k b_k(f) · t_k(x,y) + bias

数学形式：
  Branch: b = σ(W_b · f + c_b)  →  (p_dim × n_outputs)
  Trunk:  t = σ(W_t · [cos(Bx), sin(Bx)] + c_t)  →  (p_dim,)
  Output: y_k = Σ_j b_{k,j} · t_j + bias_k,  k ∈ {u, v, p}

═══════════════════════════════════════════════════════════════════
Fourier 特征编码原理
═══════════════════════════════════════════════════════════════════

标准 MLP 存在"频谱偏差"（spectral bias）：优先拟合低频分量，
对声场中的高频空间结构（涡旋细节、锐边近场）收敛缓慢。

Fourier 特征编码（Tancik et al., NeurIPS 2020）通过随机投影将输入
映射到高维 Fourier 空间，打破低频偏差：
  γ(x) = [cos(2π·B·x), sin(2π·B·x)]
其中 B 为随机频率矩阵，scale 控制频率范围。

本模型使用自适应多尺度 Fourier 特征（AMFF 思想）：
  B ~ N(0, σ²·I)，σ = fourier_scale
  较大的 fourier_scale 可覆盖更高的空间频率。
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import torch                                     # PyTorch 核心
import torch.nn as nn                            # 神经网络模块
from config import device                        # 计算设备

# 默认 DeepONet 基函数维度（Branch 和 Trunk 的公共维度）
P_DIM = 128


# ═══════════════════════════════════════════════════════════════════
# 残差块（ResidualBlock）
# ═══════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """
    残差连接块（两层全连接 + skip connection）。

    数学表达：
      h = activation(W₂ · activation(W₁ · x + b₁) + b₂) + x
                                                          ↑ skip connection

    残差连接的优势：
      1. 缓解深层网络的梯度消失问题
      2. 保证信息的恒等传播路径
      3. 使网络更容易学习"修正量"而非完整映射

    参数:
        dim:        输入/输出维度（必须相同以实现恒等连接）
        activation: 激活函数类（默认 Tanh，有界且光滑，适合物理量预测）
    """

    def __init__(self, dim: int, activation=nn.Tanh):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),                  # 第一层线性变换
            activation(),                          # 激活函数
            nn.Linear(dim, dim),                   # 第二层线性变换
        )
        self.act = activation()                    # 残差后的激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：residual + skip connection。"""
        return self.act(self.block(x) + x)         # f(x) + x，然后激活


# ═══════════════════════════════════════════════════════════════════
# Branch 网络（编码输入函数——频率参数）
# ═══════════════════════════════════════════════════════════════════
class BranchNet(nn.Module):
    """
    Branch 网络：编码驱动频率 f → 系数向量。

    物理意义：
      不同超声频率 f 会产生不同的声流模式（涡旋结构、强度、空间分布）。
      Branch Net 学习频率到声流模式系数的映射。

    数学表达：
      输入：freq_norm ∈ R¹（归一化后的频率标量）
      输出：B ∈ R^{n_outputs × p_dim}（每个输出变量对应 p_dim 个系数）
      过程：freq → MLP → reshape → (n_outputs, p_dim)

    网络结构：
      Linear(1, hidden) → Tanh → [Linear(hidden, hidden) → Tanh] × (n_layers-1) → Linear(hidden, p_dim × n_outputs)

    参数:
        p_dim:      DeepONet 基函数维度（默认 128）
        n_outputs:  输出物理量个数（u, v, p = 3）
        hidden_dim: 隐藏层宽度（默认 64，输入维度低无需太宽）
        n_layers:   隐藏层数（默认 4）
    """

    def __init__(self, p_dim: int = P_DIM, n_outputs: int = 3,
                 hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.n_outputs = n_outputs                 # 输出变量个数
        self.p_dim     = p_dim                     # 基函数维度

        # 构建 MLP 层序列
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]  # 输入层：1 → hidden
        for _ in range(n_layers - 1):                     # 中间层
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, p_dim * n_outputs))  # 输出层
        self.net = nn.Sequential(*layers)

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            freq: (N, 1) 归一化频率
        返回:
            (N, n_outputs, p_dim) 各输出变量的系数矩阵
        """
        out = self.net(freq)                       # (N, p_dim * n_outputs)
        return out.view(-1, self.n_outputs, self.p_dim)  # reshape


# ═══════════════════════════════════════════════════════════════════
# Trunk 网络（编码输出位置——空间坐标）
# ═══════════════════════════════════════════════════════════════════
class TrunkNet(nn.Module):
    """
    Trunk 网络：编码空间坐标 (x, y) → 基函数向量。

    物理意义：
      Trunk Net 学习一组空间基函数 {φ_k(x, y)}，这些基函数
      描述声流场的空间结构模式（如涡旋位置、对称性、衰减趋势）。

    数学表达：
      输入：(x, y) ∈ R²
      Fourier 编码：γ(x,y) = [cos(B·[x,y]ᵀ), sin(B·[x,y]ᵀ)] ∈ R^{2·fourier_dim}
      网络：γ → Linear → [ResidualBlock × n_layers] → Linear → Tanh → t ∈ R^{p_dim}

    网络结构：
      FourierEncode → 投影层 → 残差块堆叠 → 输出投影 → Tanh

    参数:
        p_dim:         基函数维度（默认 128）
        hidden_dim:    隐藏层宽度（默认 256，需较宽以捕获空间细节）
        n_layers:      残差块数量（默认 6，深层网络学习精细结构）
        fourier_dim:   Fourier 特征维度（默认 64，编码后维度 = 2 × 64 = 128）
        fourier_scale: Fourier 随机频率标准差（默认 5.0，控制可学习的最高空间频率）
    """

    def __init__(self, p_dim: int = P_DIM,
                 hidden_dim: int = 256, n_layers: int = 6,
                 fourier_dim: int = 64, fourier_scale: float = 5.0):
        super().__init__()

        # Fourier 随机投影矩阵（不参与梯度更新）
        # B ∈ R^{2 × fourier_dim}，B ~ N(0, scale²)
        B = torch.randn(2, fourier_dim) * fourier_scale
        self.register_buffer('B', B)               # 注册为 buffer（随模型保存但不训练）
        enc_dim = 2 * fourier_dim                   # 编码后维度（cos + sin）

        # 输入投影层：将 Fourier 编码映射到隐藏维度
        self.input_proj = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),         # 编码维度 → 隐藏维度
            nn.Tanh()                               # 激活函数
        )

        # 残差块堆叠（深层特征提取）
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, nn.Tanh)       # 每块含 2 层 + skip
            for _ in range(n_layers)
        ])

        # 输出投影层：映射到基函数空间
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, p_dim),            # 隐藏维度 → p_dim
            nn.Tanh()                                # 有界激活（DeepONet 原文建议）
        )

    def fourier_encode(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Fourier 特征编码：(x, y) → [cos(B·xy), sin(B·xy)]。

        将二维坐标映射到高维空间，打破 MLP 的频谱偏差。
        编码后维度 = 2 × fourier_dim。

        参数:
            xy: (N, 2) 空间坐标
        返回:
            (N, 2*fourier_dim) 编码后的特征向量
        """
        proj = xy @ self.B                          # (N, 2) × (2, fourier_dim) → (N, fourier_dim)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # cos + sin 拼接

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        前向传播：空间坐标 → 基函数向量。

        参数:
            xy: (N, 2) 空间坐标
        返回:
            (N, p_dim) 基函数评估值
        """
        h = self.input_proj(self.fourier_encode(xy))  # Fourier 编码 → 投影
        for block in self.res_blocks:                  # 逐层残差块
            h = block(h)
        return self.output_proj(h)                     # 输出投影


# ═══════════════════════════════════════════════════════════════════
# DeepONet 主模型（FP-PiDON）
# ═══════════════════════════════════════════════════════════════════
class AcousticStreamingPINN(nn.Module):
    """
    Frequency-Parameterized Physics-Informed DeepONet（FP-PiDON）。

    完整名称：频率参数化物理信息 DeepONet
    这是本项目的核心模型，融合了 DeepONet 算子学习与 PINN 物理约束。

    前向计算公式：
      output_k(f, x, y) = Σ_j branch_k_j(f) · trunk_j(x, y) + bias_k

      其中 k ∈ {0, 1, 2} 分别对应输出 {u, v, p}。

    输入:
        X = [x_norm, y_norm, freq_norm]  shape (N, 3)
          - x_norm:    归一化 x 坐标
          - y_norm:    归一化 y 坐标
          - freq_norm: 归一化频率 f / f_ref

    输出:
        Y = [u, v, p]  shape (N, 3)
          - u: x 方向时均声流速度 [标准化后]
          - v: y 方向时均声流速度 [标准化后]
          - p: 时均压力场 [标准化后]
          （注：输出为标准化后的值，需通过 y_normalizer.inverse_transform 还原为物理量）

    参数:
        p_dim:         Branch 与 Trunk 共享的基函数维度
        branch_hidden: Branch Net 隐藏层宽度
        branch_layers: Branch Net 隐藏层数
        trunk_hidden:  Trunk Net 隐藏层宽度
        trunk_layers:  Trunk Net 残差块数
        fourier_dim:   Fourier 编码维度
        fourier_scale: Fourier 频率尺度
        n_outputs:     输出物理量个数（u, v, p = 3）
    """

    def __init__(self, p_dim: int = P_DIM,
                 branch_hidden: int = 64,  branch_layers: int = 4,
                 trunk_hidden:  int = 256, trunk_layers:  int = 6,
                 fourier_dim:   int = 64,  fourier_scale: float = 5.0,
                 n_outputs:     int = 3):
        super().__init__()
        # Branch Net：频率 → 系数
        self.branch = BranchNet(p_dim, n_outputs, branch_hidden, branch_layers)
        # Trunk Net：空间坐标 → 基函数
        self.trunk  = TrunkNet(p_dim, trunk_hidden, trunk_layers,
                               fourier_dim, fourier_scale)
        # 输出偏置（每个输出变量一个）
        self.bias   = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DeepONet 前向传播。

        计算过程：
          1. 分离输入：xy = x[:, :2], freq = x[:, 2:3]
          2. Branch: (N, 1) → (N, 3, p_dim) — 频率编码
          3. Trunk:  (N, 2) → (N, p_dim)    — 空间编码
          4. 内积：  Σ_j branch[k,j] * trunk[j] → (N, 3)
          5. 加偏置：output + bias

        参数:
            x: (N, 3) 输入张量 [x_norm, y_norm, freq_norm]
        返回:
            (N, 3) 输出张量 [u, v, p]（标准化后）
        """
        xy        = x[:, :2]                       # 空间坐标 (N, 2)
        freq_norm = x[:, 2:3]                      # 频率 (N, 1)

        branch_out = self.branch(freq_norm)        # (N, 3, p_dim) — 系数矩阵
        trunk_out  = self.trunk(xy)                # (N, p_dim)    — 基函数值

        # Einstein 求和：对 p_dim 维度做点积
        # branch_out[n, k, p] × trunk_out[n, p] → out[n, k]
        out = torch.einsum('nkp,np->nk', branch_out, trunk_out)

        return out + self.bias                     # 加偏置

    def count_params(self):
        """统计并打印模型可训练参数总数。"""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  可训练参数: {n:,}  (Branch + Trunk[ResidualBlock] + Bias)")
        return n


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = AcousticStreamingPINN().to(device)     # 实例化模型
    model.count_params()                           # 打印参数量

    # 测试前向传播
    dummy = torch.randn(16, 3).to(device)          # 随机输入 (16, 3)
    out   = model(dummy)                           # 前向传播
    assert out.shape == (16, 3), f"输出形状错误: {out.shape}"

    print(f"  输入: {dummy.shape} → 输出: {out.shape}")
    print(f"  输出含义: [u, v, p]")
    print("[model v5] 自检通过 ✅")
