"""
trainer.py — 训练循环与模型评估（v5 声-流-固耦合版）
================================================================
【v5 修改说明】
  1. loss 字段名从 'viscous' 改为 'momentum'（对应含压力梯度的 Stokes 方程）
  2. 评估指标新增压力场 p 的相对 L2 误差
  3. 其余逻辑与 v4 保持一致

训练策略：
  1. 分步训练（渐进式引入物理约束）：
     - 前 warmup_epochs 轮：仅数据损失 + 连续性约束（无二阶导）
     - warmup_epochs 后：引入完整 Stokes 方程残差
     → 避免训练初期高阶梯度导致的数值不稳定

  2. 自适应损失权重：
     - 数据损失和 PDE 损失的量纲差异大
     - 使用 AdaptiveLossWeights 自动调整权重比例

  3. 学习率调度：
     - CosineAnnealingLR：从 lr 余弦衰减到 lr/100
     - 保证后期精细收敛

  4. 梯度裁剪：
     - max_norm=1.0，防止梯度爆炸
"""

# ═══════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════
import numpy as np                               # 数值计算
import torch                                     # PyTorch
import torch.nn as nn                            # 神经网络
import torch.optim as optim                      # 优化器
from torch.utils.data import DataLoader, TensorDataset  # 数据加载

from config import device, params                # 设备和参数
from loss import total_loss, AdaptiveLossWeights # 损失函数


# ═══════════════════════════════════════════════════════════════════
# 训练主函数
# ═══════════════════════════════════════════════════════════════════
def train_model(model, X_train, Y_train, X_val, Y_val,
                x_norm, y_norm,
                n_epochs=2000, lr=1e-3, batch_size=256,
                lambda_continuity=0.01,
                lambda_momentum=0.001,
                lambda_boundary=0.0,
                use_adaptive_weights=True,
                warmup_epochs=200,
                save_path='checkpoint.pt'):
    """
    训练 FP-PiDON（DeepONet-PINN）模型。

    训练过程：
      1. 每个 epoch 遍历所有 mini-batch
      2. 计算总损失 = 数据损失 + 物理约束损失
      3. 反向传播 + 梯度裁剪 + 参数更新
      4. 学习率调度
      5. 验证集评估
      6. 保存最佳模型

    参数:
        model:                网络模型
        X_train, Y_train:    训练集（标准化后）
        X_val, Y_val:        验证集（标准化后）
        x_norm, y_norm:      归一化器
        n_epochs:             训练轮数
        lr:                   初始学习率
        batch_size:           mini-batch 大小
        lambda_continuity:    连续性方程权重
        lambda_momentum:      Stokes 方程权重
        lambda_boundary:      边界条件权重
        use_adaptive_weights: 是否使用自适应权重
        warmup_epochs:        预热轮数（禁用动量方程）
        save_path:            检查点保存路径
    返回:
        history: 训练历史字典
    """
    # ── 初始化自适应权重（可选）──
    adaptive_weights = None                       # 默认不启用
    extra_params     = []                         # 额外可学习参数
    if use_adaptive_weights:
        adaptive_weights = AdaptiveLossWeights(n_tasks=2).to(device)
        extra_params     = list(adaptive_weights.parameters())
        print("  ✅ 启用自适应损失权重（AdaptiveLossWeights）")

    # ── 优化器（Adam + 权重衰减）──
    optimizer = optim.Adam(
        list(model.parameters()) + extra_params,  # 模型参数 + 权重参数
        lr=lr, weight_decay=1e-5                  # L2 正则化
    )

    # ── 学习率调度器（余弦退火）──
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100
    )

    # ── 数据加载器（mini-batch 随机洗牌）──
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=True
    )

    # ── 训练历史记录 ──
    history = {
        'train_loss' : [], 'val_loss'    : [],    # 总损失
        'fluid_data' : [], 'continuity'  : [],    # 数据/连续性损失
        'momentum'   : [], 'boundary_bc' : [],    # 动量/边界损失
        'w_data'     : [], 'w_pde'       : [],    # 自适应权重
    }
    best_val_loss = float('inf')                  # 最佳验证损失
    best_weights  = None                          # 最佳模型参数

    print(f"\n开始训练（FP-PiDON v5, 输出: u, v, p）...")
    print(f"  warmup_epochs={warmup_epochs}（前 {warmup_epochs} 轮禁用动量方程）")
    print("-" * 75)

    # ── 训练主循环 ──
    for epoch in range(n_epochs):
        model.train()                              # 切换到训练模式
        if adaptive_weights is not None:
            adaptive_weights.train()

        # warmup 期间不计算二阶导数（节省显存 + 避免梯度爆炸）
        use_mom = (epoch >= warmup_epochs)

        epoch_losses = []                          # 收集每个 batch 的损失

        for X_b, Y_b in train_loader:              # 遍历 mini-batch
            optimizer.zero_grad()                  # 清零梯度

            # 计算总损失
            loss, ld = total_loss(
                model, X_b, Y_b, x_norm, y_norm,
                adaptive_weights=adaptive_weights,
                lambda_continuity=lambda_continuity,
                lambda_momentum=lambda_momentum,
                lambda_boundary=lambda_boundary,
                use_momentum=use_mom,
            )

            loss.backward()                        # 反向传播
            nn.utils.clip_grad_norm_(              # 梯度裁剪
                model.parameters(), max_norm=1.0)
            optimizer.step()                       # 参数更新
            epoch_losses.append(ld)                # 记录损失

        scheduler.step()                           # 学习率更新

        # ── 验证集评估 ──
        model.eval()                               # 切换到评估模式
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(X_val), Y_val).item()

        # ── 记录历史 ──
        def _mean(key):
            return float(np.mean([d[key] for d in epoch_losses]))

        avg = _mean('total')
        history['train_loss'].append(avg)
        history['val_loss'].append(val_loss)
        history['fluid_data'].append(_mean('fluid_data'))
        history['continuity'].append(_mean('continuity'))
        history['momentum'].append(_mean('momentum'))
        history['boundary_bc'].append(_mean('boundary_bc'))
        history['w_data'].append(_mean('w_data'))
        history['w_pde'].append(_mean('w_pde'))

        # ── 保存最佳模型 ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}

        # ── 周期性打印 ──
        if (epoch + 1) % 200 == 0:
            lr_cur  = scheduler.get_last_lr()[0]
            mom_on = "ON " if use_mom else "OFF"
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"训练:{avg:.3e} | 验证:{val_loss:.3e} | "
                  f"数据:{history['fluid_data'][-1]:.3e} | "
                  f"连续:{history['continuity'][-1]:.3e} | "
                  f"动量[{mom_on}]:{history['momentum'][-1]:.3e} | "
                  f"LR:{lr_cur:.2e}")

    # ── 恢复最佳模型 ──
    model.load_state_dict(best_weights)
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")

    # ── 保存检查点 ──
    ckpt = {
        'model_state_dict': best_weights,
        'x_norm_mean'     : x_norm.mean,
        'x_norm_std'      : x_norm.std,
        'y_norm_mean'     : y_norm.mean,
        'y_norm_std'      : y_norm.std,
        'history'         : history,
        'cases_train'     : params.cases_train,
        'cases_test'      : params.cases_test,
        'freq_ref'        : params.freq_ref,
    }
    if adaptive_weights is not None:
        ckpt['adaptive_weights'] = adaptive_weights.state_dict()
    torch.save(ckpt, save_path)
    print(f"Checkpoint 已保存: {save_path}")

    return history


# ═══════════════════════════════════════════════════════════════════
# 模型评估
# ═══════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test_norm, Y_test_norm, y_norm):
    """
    在测试集上评估模型性能。

    评估指标：各物理量的相对 L2 误差
      rel_L2 = ‖y_pred - y_true‖₂ / ‖y_true‖₂ × 100%

    参数:
        model:        训练好的模型
        X_test_norm:  标准化后的测试输入
        Y_test_norm:  标准化后的测试输出
        y_norm:       输出归一化器
    返回:
        (metrics, Y_pred_real, Y_true_real)
    """
    model.eval()                                   # 评估模式
    with torch.no_grad():
        Y_pred_real = y_norm.inverse_transform(model(X_test_norm)).cpu().numpy()
        Y_true_real = y_norm.inverse_transform(Y_test_norm).cpu().numpy()

    # v5: 输出 (u, v, p)
    field_names = ['u (x方向速度)', 'v (y方向速度)', 'p (压力场)']
    metrics = {}
    print("\n测试集评估（相对 L2 误差）：")
    print("-" * 50)
    for i, name in enumerate(field_names):
        err = (np.linalg.norm(Y_pred_real[:, i] - Y_true_real[:, i]) /
               (np.linalg.norm(Y_true_real[:, i]) + 1e-10) * 100)
        metrics[name] = err
        print(f"  {name}: {err:.2f}%")

    return metrics, Y_pred_real, Y_true_real


# ═══════════════════════════════════════════════════════════════════
# 独立调试
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_loader import load_all_data, build_train_test_tensors, build_normalizers
    from model import AcousticStreamingPINN

    train_ds, test_ds = load_all_data()
    X_train, Y_train, X_test, Y_test = build_train_test_tensors(train_ds, test_ds)
    x_norm, y_norm = build_normalizers(X_train, Y_train)

    X_train_n = x_norm.transform(X_train)
    Y_train_n = y_norm.transform(Y_train)
    X_test_n  = x_norm.transform(X_test)
    Y_test_n  = y_norm.transform(Y_test)

    model = AcousticStreamingPINN().to(device)
    model.count_params()

    history = train_model(
        model, X_train_n, Y_train_n, X_test_n, Y_test_n,
        x_norm, y_norm,
        n_epochs=10, lr=1e-3, batch_size=64, warmup_epochs=3,
        save_path='checkpoint_debug.pt'
    )
    metrics, _, _ = evaluate_model(model, X_test_n, Y_test_n, y_norm)
    print("\n[trainer v5] 自检通过 ✅")
