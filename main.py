"""
main.py — 主程序（v5 声-流-固耦合完整版）
================================================================
完整流程：声场参数 → 声流场预测 → 固体受力计算 → 微结构运动预测

物理建模核心思想：
  超声频率 ~ O(100 kHz)，固体在声场中会产生高频振动，
  但该振动时间尺度远小于声流驱动的平均运动时间尺度。
  因此模型不直接学习固体的高频振动轨迹，而是：
    1. 学习时间平均后的声流场 (u, v, p) — DeepONet 算子映射
    2. 通过物理公式计算固体受力 — 表面积分
    3. 预测其低频运动行为 — 牛顿动力学积分
"""

import os
import torch
import numpy as np

from config import params, device
from data_loader import (load_all_data, build_train_test_tensors, build_normalizers)
from model import AcousticStreamingPINN
from trainer import (train_model, evaluate_model)
from force_calculator import ForceCalculator
from dynamics import (MicrostructureDynamics, trajectory_summary)
from visualize import (plot_loss_curves, save_all_figures, plot_trajectory, plot_summary)
from network_viz import generate_all_architecture_plots


def main():
    # ── Step 0: 系统信息 ──
    print("=" * 72)
    print("  FP-PiDON v5: Acoustic-Fluid-Solid Coupling AI Surrogate")
    print("=" * 72)
    print(f"  Device:    {device}")
    print(f"  Domain:    x=[{params.Lx_min*1e6:.0f},{params.Lx_max*1e6:.0f}]um "
          f"y=[0,{params.Ly*1e6:.0f}]um")
    print(f"  Solid:     triangle (base={params.tri_base*1e6:.1f}um "
          f"h={params.tri_height*1e6:.0f}um apex={params.tri_apex_angle}deg)")
    print(f"  Input:     (x, y, freq) -> Output: (u, v, p)")
    print(f"  Train:     {[f'{c[0]/1e6:.2f}MHz' for c in params.cases_train]}")
    print(f"  Test:      {[f'{c[0]/1e6:.2f}MHz' for c in params.cases_test]}")
    print("=" * 72)

    os.makedirs('results', exist_ok=True)

    # ── Step 1: 数据加载 ──
    print("\n" + "=" * 60)
    print("  Step 1: Data Loading")
    print("=" * 60)
    train_ds, test_ds = load_all_data()
    X_train, Y_train, X_test, Y_test = build_train_test_tensors(train_ds, test_ds)
    x_norm, y_norm = build_normalizers(X_train, Y_train)
    X_train_n = x_norm.transform(X_train)
    Y_train_n = y_norm.transform(Y_train)
    X_test_n  = x_norm.transform(X_test)
    Y_test_n  = y_norm.transform(Y_test)

    # ── Step 2: 模型构建 ──
    print("\n" + "=" * 60)
    print("  Step 2: Model Construction")
    print("=" * 60)
    model = AcousticStreamingPINN(
        p_dim=128, branch_hidden=64, branch_layers=4,
        trunk_hidden=256, trunk_layers=6,
        fourier_dim=64, fourier_scale=5.0, n_outputs=3,
    ).to(device)
    model.count_params()

    # ── Step 3: 模型训练 ──
    print("\n" + "=" * 60)
    print("  Step 3: Model Training")
    print("=" * 60)
    history = train_model(
        model, X_train_n, Y_train_n, X_test_n, Y_test_n,
        x_norm, y_norm,
        n_epochs=2000, lr=1e-3, batch_size=256,
        lambda_continuity=0.01, lambda_momentum=0.001,
        lambda_boundary=0.0, use_adaptive_weights=True,
        warmup_epochs=300, save_path='checkpoint_v5.pt',
    )

    # ── Step 4: 流场预测评估 ──
    print("\n" + "=" * 60)
    print("  Step 4: Flow Field Evaluation")
    print("=" * 60)
    metrics, Y_pred_real, Y_true_real = evaluate_model(
        model, X_test_n, Y_test_n, y_norm
    )

    model.eval()
    with torch.no_grad():
        freq_new = 7.5e5
        pt = torch.FloatTensor([[0.5, 0.5, freq_new / params.freq_ref]]).to(device)
        pred = y_norm.inverse_transform(model(x_norm.transform(pt))).cpu().numpy()
    print(f"\n  Inference @ {freq_new/1e6:.2f}MHz, center:")
    print(f"    u={pred[0,0]:.4e} m/s, v={pred[0,1]:.4e} m/s, p={pred[0,2]:.4e} Pa")

    # ── Step 5: 固体受力 ──
    print("\n" + "=" * 60)
    print("  Step 5: Solid Force Computation")
    print("=" * 60)
    force_calc = ForceCalculator(model, x_norm, y_norm,
                                 n_per_edge=params.n_surface_pts // 3)

    print(f"\n  {'freq[MHz]':>10}  {'Fx[N/m]':>14}  {'Fy[N/m]':>14}  "
          f"{'Fx+ARF':>14}  {'Fy+ARF':>14}")
    print("  " + "-" * 68)
    for freq in [3e5, 5e5, 7e5, 1e6]:
        Fx, Fy = force_calc.compute_forces(freq)
        Fx_t, Fy_t = force_calc.compute_forces_with_arf(freq)
        print(f"  {freq/1e6:10.2f}  {Fx:14.4e}  {Fy:14.4e}  "
              f"{Fx_t:14.4e}  {Fy_t:14.4e}")

    # ── Step 6: 动力学仿真 ──
    print("\n" + "=" * 60)
    print("  Step 6: Microstructure Dynamics")
    print("=" * 60)
    sim_freq = 5e5
    dyn = MicrostructureDynamics(
        force_calculator=force_calc, freq=sim_freq,
        dt=params.dt_dynamics, use_arf=True,
    )

    n_demo = min(50, params.n_steps)
    print(f"\n  Euler integration ({n_demo} steps):")
    traj_euler = dyn.simulate_euler(x0=0, y0=0, vx0=0, vy0=0, n_steps=n_demo)
    trajectory_summary(traj_euler)

    print(f"\n  RK4 integration ({n_demo} steps):")
    traj_rk4 = dyn.simulate_rk4(x0=0, y0=0, vx0=0, vy0=0, n_steps=n_demo)
    trajectory_summary(traj_rk4)

    # ── Step 7: 可视化 ──
    print("\n" + "=" * 60)
    print("  Step 7: Visualization")
    print("=" * 60)
    plot_loss_curves(history, save_path='results/loss_curves.png')

    test_freq_mhz = params.cases_test[0][0] / 1e6
    save_all_figures(X_test, Y_pred_real, Y_true_real,
                     freq_mhz=test_freq_mhz, out_dir='results')
    plot_summary(X_test, Y_pred_real, Y_true_real, history,
                 freq_mhz=test_freq_mhz, save_path='results/summary.png')
    plot_trajectory(traj_euler, save_path='results/trajectory_euler.png')
    plot_trajectory(traj_rk4,   save_path='results/trajectory_rk4.png')
    generate_all_architecture_plots(model=model, out_dir='results')

    # ── 完成 ──
    print("\n" + "=" * 72)
    print("  Pipeline complete! Results saved to ./results/")
    print("=" * 72)
    print("\n  Complete workflow:")
    print("    freq (acoustic params)")
    print("      |  [Branch Net]")
    print("    DeepONet operator mapping")
    print("      |  [Trunk Net + Physics constraints]")
    print("    (u, v, p) streaming field")
    print("      |  [Surface integral]")
    print("    (Fx, Fy) fluid forces")
    print("      |  [Newton dynamics + time integration]")
    print("    x(t), y(t) microstructure trajectory")

    if os.path.isdir('results'):
        print(f"\n  Generated files:")
        for f in sorted(os.listdir('results')):
            sz = os.path.getsize(os.path.join('results', f)) / 1024
            print(f"    {f:45s} {sz:8.1f} KB")


if __name__ == "__main__":
    main()
