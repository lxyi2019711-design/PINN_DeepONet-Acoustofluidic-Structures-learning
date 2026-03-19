"""
data_loader.py — 数据加载、预处理与归一化（v5 声-流-固耦合版）
================================================================
【v5 升级说明】
  1. CSV 格式扩展为 6 列：x, y, u, v, p, velocity（或 x, y, u, v, p）
  2. 输出 Y 变更为 (u, v, p)：x方向速度、y方向速度、时均压力
  3. 合成数据新增压力场合成（基于 Bernoulli 近似）
  4. 保留原有归一化器 Normalizer 和数据流水线

COMSOL CSV 导出格式（SI 单位，第一行标题）：
  x[m], y[m], u[m/s], v[m/s], p[Pa]

COMSOL 导出步骤：
  Results → Export → Data（选择流体域）
  表达式：x, y, spf.U, spf.V, spf.p
  格式：CSV，勾选列标题

数据处理流程：
  1. 读取 CSV → 过滤三角形固体域内的点 → 归一化坐标
  2. 构造输入 X = (x_norm, y_norm, freq_norm)，输出 Y = (u, v, p)
  3. 标准化：(x - mean) / std，支持正反变换
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import os                                        # 文件路径操作
import numpy as np                               # 数值计算
import torch                                     # PyTorch 张量
from config import params, device, COMSOL_DATA_DIR  # 全局配置
from geometry import in_fluid_domain, sample_fluid_points  # 几何工具


# ═══════════════════════════════════════════════════════════════════
# 第一部分：文件路径工具
# ═══════════════════════════════════════════════════════════════════
def _case_filename(split: str, idx: int) -> str:
    """
    生成 COMSOL CSV 文件的完整路径。
    命名规则：data/case_train_0.csv, data/case_test_1.csv 等。

    参数:
        split: 'train' 或 'test'（数据集划分）
        idx:   工况编号（从 0 开始）
    返回:
        文件完整路径字符串
    """
    return os.path.join(COMSOL_DATA_DIR, f"case_{split}_{idx}.csv")


# ═══════════════════════════════════════════════════════════════════
# 第二部分：读取单个 COMSOL CSV 文件
# ═══════════════════════════════════════════════════════════════════
def load_comsol_csv(filepath: str, case_params: tuple) -> dict:
    """
    读取单个 COMSOL CSV 文件，解析物理场数据。

    CSV 列定义（v5 版本）：
      列 0: x [m]       — x 坐标
      列 1: y [m]       — y 坐标
      列 2: u [m/s]     — x 方向时均声流速度
      列 3: v [m/s]     — y 方向时均声流速度
      列 4: p [Pa]      — 时均压力场

    处理步骤：
      1. 读取原始数据，处理 NaN 值
      2. 过滤三角形固体域内的点（仅保留流体域数据）
      3. 归一化坐标到 [0, 1] 区间

    参数:
        filepath:    CSV 文件路径
        case_params: (freq,) 工况参数元组
    返回:
        dict 字典，包含归一化坐标和物理场量
    """
    (freq,) = case_params                        # 解包频率参数

    # 读取 CSV（跳过标题行，逗号分隔）
    raw = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    if raw.ndim == 1:                            # 处理单行数据
        raw = raw[np.newaxis, :]
    raw = np.where(np.isnan(raw), 0.0, raw)      # NaN 替换为 0

    # 解析各列
    x_m = raw[:, 0]                              # x 坐标 [m]
    y_m = raw[:, 1]                              # y 坐标 [m]
    u   = raw[:, 2]                              # u 速度 [m/s]
    v   = raw[:, 3]                              # v 速度 [m/s]
    p   = raw[:, 4]                              # 压力 [Pa]

    # 过滤三角形固体域内的点（固体域数据不参与训练）
    mask = in_fluid_domain(x_m, y_m)             # 布尔掩码
    x_m, y_m, u, v, p = x_m[mask], y_m[mask], u[mask], v[mask], p[mask]

    print(f"    过滤后有效点数: {mask.sum()} / {len(mask)}")

    return {
        'x'   : (x_m - params.Lx_min) / params.Lx,  # 归一化 x → [0, 1]
        'y'   : y_m / params.Ly,                      # 归一化 y → [0, 1]
        'u'   : u,                                     # x 方向速度 [m/s]
        'v'   : v,                                     # y 方向速度 [m/s]
        'p'   : p,                                     # 压力 [Pa]
        'freq': freq,                                  # 驱动频率 [Hz]
    }


# ═══════════════════════════════════════════════════════════════════
# 第三部分：批量加载 COMSOL 数据
# ═══════════════════════════════════════════════════════════════════
def load_all_comsol_data() -> tuple:
    """
    批量加载所有训练和测试 CSV 文件。
    文件命名规则：case_train_0.csv, case_train_1.csv, ...

    返回:
        (train_datasets, test_data) — 训练集列表和测试集字典
    """
    # 加载训练数据
    train_datasets = []
    for idx, case in enumerate(params.cases_train):
        fp = _case_filename('train', idx)         # 生成文件路径
        if not os.path.exists(fp):                # 文件存在性检查
            raise FileNotFoundError(
                f"\n找不到训练文件：{fp}\n"
                f"请将 COMSOL 导出 CSV 放入 {COMSOL_DATA_DIR}/\n"
                f"格式：x, y, u, v, p（5列，SI 单位）"
            )
        data = load_comsol_csv(fp, case)          # 读取单个文件
        train_datasets.append(data)
        print(f"  ✅ 训练[{idx}] freq={case[0]/1e6:.2f}MHz → {len(data['x'])} 点")

    # 加载测试数据
    test_datasets = []
    for idx, case in enumerate(params.cases_test):
        fp = _case_filename('test', idx)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"找不到测试文件：{fp}")
        data = load_comsol_csv(fp, case)
        test_datasets.append(data)
        print(f"  ✅ 测试[{idx}] freq={case[0]/1e6:.2f}MHz → {len(data['x'])} 点")

    # 合并测试数据（如果有多个测试工况）
    test_data = (test_datasets[0] if len(test_datasets) == 1
                 else _merge_datasets(test_datasets))
    return train_datasets, test_data


def _merge_datasets(datasets: list) -> dict:
    """
    合并多个数据集字典（将各工况的点云拼接在一起）。

    参数:
        datasets: 字典列表
    返回:
        合并后的字典
    """
    merged = {}
    for key in ['x', 'y', 'u', 'v', 'p']:       # 拼接所有物理量
        merged[key] = np.concatenate([d[key] for d in datasets])
    merged['freq'] = datasets[0]['freq']          # 频率取第一个工况
    return merged


# ═══════════════════════════════════════════════════════════════════
# 第四部分：合成数据（调试用，无 COMSOL 数据时自动启用）
# ═══════════════════════════════════════════════════════════════════
def generate_synthetic_data(case_params: tuple, n_points: int = 800) -> dict:
    """
    生成合成声流场数据，仅用于代码调试和验证流程正确性。
    不代表真实物理，但保证数据格式与 COMSOL 导出一致。

    合成模型：
      声致衰减系数 α = 8π²μf² / (3ρc³)  [1/m]
      u(x,y) ∝ sin(πy) · (1-e^{-5x}) · e^{-αx}     （x 方向声流速度）
      v(x,y) ∝ cos(2πx) · sin(πy)                     （y 方向，回流分量）
      p(x,y) ∝ -½ρ(u²+v²) + P0·cos(2πx)·sin(πy)     （Bernoulli 近似 + 声压项）

    参数:
        case_params: (freq,) 频率元组
        n_points:    采样点数
    返回:
        dict 字典（格式同 COMSOL 数据）
    """
    (freq,) = case_params                        # 解包频率

    # 在流体域内随机采样（排除三角形）
    x_m, y_m = sample_fluid_points(n_points, seed=int(freq))
    x_n = (x_m - params.Lx_min) / params.Lx     # 归一化 x → [0, 1]
    y_n = y_m / params.Ly                        # 归一化 y → [0, 1]

    # 声致衰减系数 α [1/m]（声能在粘性流体中的衰减率）
    alpha   = (8 * np.pi**2 * params.mu * freq**2 /
               (3 * params.rho_f * params.c_sound**3))
    alpha_n = alpha * params.Lx                   # 归一化域上的衰减量

    # 合成 x 方向声流速度 [m/s]
    u = (1e-4 * np.sin(np.pi * y_n) *
         (1 - np.exp(-5 * x_n)) * np.exp(-alpha_n * x_n))

    # 合成 y 方向声流速度 [m/s]（回流分量，量级较小）
    v = (2e-5 * np.cos(2 * np.pi * x_n) * np.sin(np.pi * y_n))

    # 合成压力场 [Pa]（Bernoulli 近似 + 声压驻波项）
    p = (-0.5 * params.rho_f * (u**2 + v**2) +
         100.0 * np.cos(2 * np.pi * x_n) * np.sin(np.pi * y_n))

    return {
        'x'   : x_n,                             # 归一化 x 坐标
        'y'   : y_n,                              # 归一化 y 坐标
        'u'   : u,                                # x 方向速度 [m/s]
        'v'   : v,                                # y 方向速度 [m/s]
        'p'   : p,                                # 压力 [Pa]
        'freq': freq,                             # 频率 [Hz]
    }


# ═══════════════════════════════════════════════════════════════════
# 第五部分：统一数据加载入口
# ═══════════════════════════════════════════════════════════════════
def load_all_data() -> tuple:
    """
    自动判断数据来源：若存在 COMSOL 数据目录则加载真实数据，
    否则生成合成数据（调试模式）。

    返回:
        (train_datasets, test_data) — 训练集列表和测试集字典
    """
    print(f"\n正在加载数据（{len(params.cases_train)} 个频率工况）...")
    if os.path.isdir(COMSOL_DATA_DIR):           # 检查数据目录
        print(f"  检测到 {COMSOL_DATA_DIR}，加载 COMSOL 数据")
        return load_all_comsol_data()
    else:
        print(f"  未找到 {COMSOL_DATA_DIR}，使用合成数据（调试模式）")
        train_datasets = []
        for i, case in enumerate(params.cases_train):
            data = generate_synthetic_data(case, n_points=800)
            train_datasets.append(data)
            print(f"  训练[{i}]: {case[0]/1e6:.1f}MHz → {len(data['x'])} 点")
        test_data = generate_synthetic_data(params.cases_test[0], n_points=400)
        print(f"  测试: {params.cases_test[0][0]/1e6:.2f}MHz → {len(test_data['x'])} 点")
        return train_datasets, test_data


# ═══════════════════════════════════════════════════════════════════
# 第六部分：字典 → PyTorch 张量转换
# ═══════════════════════════════════════════════════════════════════
def prepare_tensors(dataset: dict) -> tuple:
    """
    将单个工况的字典数据转换为 PyTorch 张量。

    输入张量 X: (N, 3) = [x_norm, y_norm, freq_norm]
      - x_norm:    归一化 x 坐标，∈ [0, 1]
      - y_norm:    归一化 y 坐标，∈ [0, 1]
      - freq_norm: 归一化频率，freq / freq_ref

    输出张量 Y: (N, 3) = [u, v, p]
      - u: x 方向时均声流速度 [m/s]
      - v: y 方向时均声流速度 [m/s]
      - p: 时均压力 [Pa]

    参数:
        dataset: 数据字典
    返回:
        (X, Y) PyTorch 张量元组，已移至目标设备
    """
    n        = len(dataset['x'])                  # 数据点数
    freq_col = np.full(n, dataset['freq'] / params.freq_ref)  # 归一化频率列

    # 构造输入矩阵 (N, 3)
    X = np.column_stack([dataset['x'], dataset['y'], freq_col])
    # 构造输出矩阵 (N, 3) — v5: (u, v, p)
    Y = np.column_stack([dataset['u'], dataset['v'], dataset['p']])

    return (torch.FloatTensor(X).to(device),      # 转为 GPU 张量
            torch.FloatTensor(Y).to(device))


def build_train_test_tensors(train_datasets: list, test_data: dict) -> tuple:
    """
    将训练集列表和测试集字典转换为拼接后的大张量。

    返回:
        (X_train, Y_train, X_test, Y_test) — 4 个张量
    """
    X_list, Y_list = [], []                       # 存储各工况张量
    for d in train_datasets:
        X, Y = prepare_tensors(d)                 # 转换单个工况
        X_list.append(X)
        Y_list.append(Y)

    X_train = torch.cat(X_list, dim=0)            # 拼接所有训练数据
    Y_train = torch.cat(Y_list, dim=0)
    X_test, Y_test = prepare_tensors(test_data)   # 转换测试数据

    # 打印数据统计信息
    print(f"\n训练集: X={X_train.shape}, Y={Y_train.shape}")
    print(f"测试集: X={X_test.shape},  Y={Y_test.shape}")
    labels = ['x_norm', 'y_norm', 'freq_norm']
    for i, label in enumerate(labels):
        col = X_train[:, i]
        print(f"  {label}: [{col.min():.3f}, {col.max():.3f}]")
    return X_train, Y_train, X_test, Y_test


# ═══════════════════════════════════════════════════════════════════
# 第七部分：数据归一化器
# ═══════════════════════════════════════════════════════════════════
class Normalizer:
    """
    标准化工具：z = (x - mean) / std

    用于将输入/输出数据标准化到零均值、单位方差，
    改善神经网络训练的数值稳定性和收敛速度。

    方法:
      fit(data):              计算均值和标准差
      transform(data):        正向变换 z = (x - μ) / σ
      inverse_transform(data): 反向变换 x = z × σ + μ
      save(path) / load(path): 保存/加载归一化参数
    """

    def __init__(self):
        self.mean = None                          # 均值（训练时计算）
        self.std  = None                          # 标准差（训练时计算）

    def fit(self, data: torch.Tensor):
        """从数据中计算归一化参数（均值和标准差）。"""
        self.mean = data.mean(dim=0, keepdim=True)   # 各列均值
        self.std  = data.std(dim=0, keepdim=True)    # 各列标准差
        self.std[self.std < 1e-8] = 1.0              # 防止除零
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """正向标准化：z = (x - μ) / σ"""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """反向还原：x = z × σ + μ"""
        return data * self.std + self.mean

    def save(self, path: str):
        """保存归一化参数到文件。"""
        torch.save({'mean': self.mean, 'std': self.std}, path)

    @classmethod
    def load(cls, path: str):
        """从文件加载归一化参数。"""
        ck  = torch.load(path, map_location='cpu')
        obj = cls()
        obj.mean = ck['mean']
        obj.std  = ck['std']
        return obj


def build_normalizers(X_train: torch.Tensor,
                      Y_train: torch.Tensor) -> tuple:
    """
    根据训练数据构建输入/输出归一化器。

    返回:
        (x_normalizer, y_normalizer) — 输入和输出的归一化器
    """
    x_normalizer = Normalizer().fit(X_train)      # 输入归一化
    y_normalizer = Normalizer().fit(Y_train)      # 输出归一化
    return x_normalizer, y_normalizer


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_ds, test_ds = load_all_data()
    X_train, Y_train, X_test, Y_test = build_train_test_tensors(train_ds, test_ds)
    x_norm, y_norm = build_normalizers(X_train, Y_train)
    print(f"\n[data_loader v5] 自检通过 ✅")
    print(f"  输入维度: {X_train.shape[1]}  (x, y, freq)")
    print(f"  输出维度: {Y_train.shape[1]}  (u, v, p)")
    print(f"  Y_train 输出范围:")
    for i, name in enumerate(['u [m/s]', 'v [m/s]', 'p [Pa]']):
        print(f"    {name}: [{Y_train[:,i].min():.4e}, {Y_train[:,i].max():.4e}]")
