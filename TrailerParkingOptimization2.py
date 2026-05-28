"""
半挂牵引车泊车轨迹优化

CasADi + Ipopt 求解自由终端时间最优控制问题。
半挂牵引车模型，梯形积分离散，圆形避障约束。

状态量: [x, y, theta, psi, v]  (牵引车位置、牵引车朝向、挂车朝向、速度)
控制量: [delta, a]              (前轮转向角, 加速度)
目标:   min( w_T * T + w_u * integral(delta^2 + a^2) )
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import DualMultWS

# ==================== 参数 ====================
N = 80                       # 离散区间数
L = 4.6                      # 牵引车轴距 [m]
W = 2.5                      # 轴宽
L_TRAILER = 10.0              # 铰接点到挂车车轴距离 [m]

# 状态范围
X_BOUNDS = (-25.0, 25.0)
Y_BOUNDS = (-10.0, 15.0)
THETA_BOUNDS = (-4 * np.pi, 4 * np.pi)
PSI_BOUNDS = THETA_BOUNDS
V_BOUNDS = (-4.0, 4.0)      # 允许倒车

l_BOUNDS = (0.0, ca.inf)
n_BOUNDS = (0.0, ca.inf)
sl_BOUNDS = (0.0, ca.inf)

# 控制范围
DELTA_BOUNDS = (np.deg2rad(-35), np.deg2rad(35))
A_BOUNDS = (-10.0, 10.0)

# 时间步长范围
DT_BOUNDS = (0.2, 0.3)
T_GUESS = 15.0

# 初始 / 终端状态 [x, y, theta, psi, v]
STATE_0 = (-10.0, 7.5, 0.0, 0.0, 2.0)
# STATE_0 = (9.841, 7.923, -0.1458, 0.1482, -1.0)
STATE_F = (0.0, -1.0, np.pi / 2, np.pi / 2, 0)

# 障碍物 (使用 obstHrep 计算的 A, b)
nOb = 3                                          # 障碍物数量
vOb = [4, 4, 4]                                        # 每个障碍物的边数
total_edges = sum(vOb)                           # 总边数
A = np.array([[0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0]])
b = np.array([25.0, 25.0, -15.0, 25.0,  # 上
              4.5, -2.5, 12.5, 25.0,   # 左下
              4.5, 25.0, 12.5, -2.5    # 右下
              ])
d_min = 0.25

# 目标函数权重
W_TIME = 0.0
W_CTRL = 0.05
W_BETA = 0.1


# ==================== 动力学模型 ====================
def trailer_ode(x, y, theta, psi, v, delta, a):
    """半挂牵引车模型状态导数"""
    beta = theta - psi
    return (
        v * ca.cos(theta),          # xdot
        v * ca.sin(theta),          # ydot
        v * ca.tan(delta) / L,      # thetadot (牵引车)
        v * ca.sin(beta) / L_TRAILER,  # psidot (挂车)
        a,                          # vdot
    )


# ==================== 构建 NLP ====================
def build_problem():
    # ---- 决策变量 ----
    X     = ca.SX.sym("X",     N + 1)
    Y     = ca.SX.sym("Y",     N + 1)
    Theta = ca.SX.sym("Theta", N + 1)
    Psi   = ca.SX.sym("Psi",   N + 1)
    V     = ca.SX.sym("V",     N + 1)
    l     = ca.SX.sym("lambda", 2 * total_edges, N + 1)   # 前一半挂车，后一半牵引车
    n     = ca.SX.sym("miu",    8 * nOb, N + 1)            # 每个障碍物独立: [tr_j, tc_j], 每组4个
    sl    = ca.SX.sym("sl",     2 * nOb, N + 1)            # 前nOb挂车，后nOb牵引车
    Delta = ca.SX.sym("Delta", N)
    Acc   = ca.SX.sym("Acc",   N)
    DT    = ca.SX.sym("DT",    N)

    w = ca.vertcat(X, Y, Theta, Psi, V, l.reshape((-1, 1)), n.reshape((-1, 1)), sl.reshape((-1, 1)), Delta, Acc, DT)

    # ---- 目标函数 ----
    total_time = ca.sum1(DT)
    ctrl_cost  = ca.sum1(Delta**2) + 0.01*ca.sum1(Acc**2)
    beta_cost = ca.sum1((Theta-Psi)**2)
    objective  = W_TIME * total_time + W_CTRL * ctrl_cost + W_BETA*beta_cost + 100*ca.sum(sl**2) + 1000*ca.sum(sl)
    # objective = 10*ca.sum(sl**2) + 100*ca.sum(sl)
    # ---- 约束 ----
    g, lbg, ubg = [], [], []

    # 动力学约束 — 梯形积分
    for k in range(N):
        dt_k = DT[k]
        f0 = trailer_ode(X[k],   Y[k],   Theta[k],   Psi[k],   V[k],   Delta[k], Acc[k])
        f1 = trailer_ode(X[k+1], Y[k+1], Theta[k+1], Psi[k+1], V[k+1], Delta[k], Acc[k])
        sk  = (X[k],   Y[k],   Theta[k],   Psi[k],   V[k])
        sk1 = (X[k+1], Y[k+1], Theta[k+1], Psi[k+1], V[k+1])
        for s0, s1, fd0, fd1 in zip(sk, sk1, f0, f1):
            g.append(s1 - s0 - 0.5 * dt_k * (fd0 + fd1))
            lbg.append(0.0); ubg.append(0.0)
        # DT约束
        # g.append(0.05+0.2*V[k]**2-DT[k]);lbg.append(0.0); ubg.append(100.0)

    # 初始状态约束
    for var, val in zip((X, Y, Theta, Psi, V), STATE_0):
        g.append(var[0] - val); lbg.append(0.0); ubg.append(0.0)

    # 终端状态约束
    for var, val in zip((X, Y, Theta, Psi, V), STATE_F):
        g.append(var[N] - val); lbg.append(0.0); ubg.append(0.0)

    # 车身几何参数
    g_b_trailer = ca.DM([L_TRAILER / 2, W / 2, L_TRAILER / 2, W / 2])
    g_b_tractor = ca.DM([L / 2, W / 2, L / 2, W / 2])
    offset_trailer = L_TRAILER / 2
    offset_tractor = L / 2

    # 按障碍物划分 A, b 的索引偏移
    edge_offset = [0]
    for nv in vOb:
        edge_offset.append(edge_offset[-1] + nv)

    for k in range(N + 1):
        for j in range(nOb):
            e_start = edge_offset[j]
            e_end   = edge_offset[j + 1]
            Aj = A[e_start:e_end, :]
            bj = b[e_start:e_end]

            # ========== 挂车 (朝向 Psi) ==========
            # norm(A'*lambda) == 1
            l_tr = l[e_start:e_end, k]
            V_x = ca.dot(Aj[:, 0], l_tr)
            V_y = ca.dot(Aj[:, 1], l_tr)
            g.append(V_x**2 + V_y**2);lbg.append(1.0);ubg.append(1.0)

            # G'*mu + R'*A*lambda = 0
            n_tr_j = n[4*j:4*j+4, k]
            x_c = n_tr_j[0] - n_tr_j[2] + ca.cos(Psi[k])*V_x + ca.sin(Psi[k])*V_y
            y_c = n_tr_j[1] - n_tr_j[3] - ca.sin(Psi[k])*V_x + ca.cos(Psi[k])*V_y
            g.append(x_c);lbg.append(0.0);ubg.append(0.0)
            g.append(y_c);lbg.append(0.0);ubg.append(0.0)

            # 挂车参考点: (X, Y) - cos/sin(Psi)*offset_trailer
            dist = (-ca.dot(g_b_trailer, n_tr_j) +
                    (X[k] - ca.cos(Psi[k])*offset_trailer)*V_x +
                    (Y[k] - ca.sin(Psi[k])*offset_trailer)*V_y -
                    ca.dot(bj, l_tr) + sl[j, k])
            g.append(dist);lbg.append(d_min);ubg.append(1000.0)

            # ========== 牵引车 (朝向 Theta) ==========
            l_tc = l[total_edges + e_start:total_edges + e_end, k]
            V_x_t = ca.dot(Aj[:, 0], l_tc)
            V_y_t = ca.dot(Aj[:, 1], l_tc)
            g.append(V_x_t**2 + V_y_t**2);lbg.append(1.0);ubg.append(1.0)

            n_tc_j = n[4*nOb + 4*j:4*nOb + 4*j+4, k]
            x_c_t = n_tc_j[0] - n_tc_j[2] + ca.cos(Theta[k])*V_x_t + ca.sin(Theta[k])*V_y_t
            y_c_t = n_tc_j[1] - n_tc_j[3] - ca.sin(Theta[k])*V_x_t + ca.cos(Theta[k])*V_y_t
            g.append(x_c_t);lbg.append(0.0);ubg.append(0.0)
            g.append(y_c_t);lbg.append(0.0);ubg.append(0.0)

            # 牵引车参考点: (X, Y) + cos/sin(Theta)*offset_tractor
            dist_t = (-ca.dot(g_b_tractor, n_tc_j) +
                      (X[k] + ca.cos(Theta[k])*offset_tractor)*V_x_t +
                      (Y[k] + ca.sin(Theta[k])*offset_tractor)*V_y_t -
                      ca.dot(bj, l_tc) + sl[nOb + j, k])
            g.append(dist_t);lbg.append(d_min);ubg.append(1000.0)

    # ---- 变量边界 ----
    lbx = (
        [X_BOUNDS[0]]     * (N+1) + [Y_BOUNDS[0]]     * (N+1) +
        [THETA_BOUNDS[0]] * (N+1) + [PSI_BOUNDS[0]]   * (N+1) +
        [V_BOUNDS[0]]     * (N+1) + [l_BOUNDS[0]]     * (2 * total_edges * (N+1)) +
        [n_BOUNDS[0]]     * (8 * nOb * (N+1)) + [sl_BOUNDS[0]] * (2 * nOb * (N+1)) +
        [DELTA_BOUNDS[0]] * N + [A_BOUNDS[0]]         * N +
        [DT_BOUNDS[0]]    * N
    )
    ubx = (
        [X_BOUNDS[1]]     * (N+1) + [Y_BOUNDS[1]]     * (N+1) +
        [THETA_BOUNDS[1]] * (N+1) + [PSI_BOUNDS[1]]   * (N+1) +
        [V_BOUNDS[1]]     * (N+1) + [l_BOUNDS[1]]     * (2 * total_edges * (N+1)) +
        [n_BOUNDS[1]]     * (8 * nOb * (N+1)) + [sl_BOUNDS[1]] * (2 * nOb * (N+1)) +
        [DELTA_BOUNDS[1]] * N + [A_BOUNDS[1]]         * N +
        [DT_BOUNDS[1]]    * N
    )

    nlp = {"x": w, "f": objective, "g": ca.vertcat(*g)}
    return nlp, lbx, ubx, lbg, ubg


# ==================== 初始猜测 ====================
def initial_guess():
    # """起点到终点的线性插值作为初值"""
    # s = np.linspace(0.0, 1.0, N + 1)
    #
    # def lerp(start, end):
    #     return start + (end - start) * s
    #
    # X_g     = lerp(STATE_0[0], STATE_F[0])
    # Y_g     = lerp(STATE_0[1], STATE_F[1])
    # Theta_g = lerp(STATE_0[2], STATE_F[2])
    # Psi_g   = lerp(STATE_0[3], STATE_F[3])
    # V_g     = lerp(STATE_0[4], STATE_F[4])
    #
    # Delta_g = np.zeros(N)
    # Acc_g   = np.zeros(N)
    # DT_g    = np.full(N, T_GUESS / N)
    data = np.load('traj2.npz', allow_pickle=True)
    traj = data['traj'].item()
    N_ref = len(traj['Delta'])

    def interp(arr, N_out, is_state=False):
        """将参考轨迹从 N_ref 线性插值到 N 个点。
        is_state=True: arr 长度为 N_ref+1（状态量在节点上）
        is_state=False: arr 长度为 N_ref（控制量/步长在区间上）
        """
        if N_ref == N:
            return arr
        s_old = np.linspace(0.0, 1.0, len(arr))
        s_new = np.linspace(0.0, 1.0, N_out if not is_state else N_out + 1)
        return np.interp(s_new, s_old, arr)

    X_g     = interp(traj['X'],     N, is_state=True)
    Y_g     = interp(traj['Y'],     N, is_state=True)
    Theta_g = interp(traj['Theta'], N, is_state=True)
    Psi_g   = interp(traj['Psi'],   N, is_state=True)
    V_g     = interp(traj['V'],     N, is_state=True)

    Delta_g = interp(traj['Delta'], N)
    Acc_g   = interp(traj['Acc'],   N)
    DT_g    = interp(traj['DT'],    N)

    # 挂车 warm-start
    g_b_tr = ca.DM([L_TRAILER / 2, W / 2, L_TRAILER / 2, W / 2])
    l_tr, n_tr, d_tr = DualMultWS.solve(A, b, nOb, vOb, N, X_g, Y_g, Psi_g,
                                        g_b_tr, L_TRAILER / 2, -1)
    # 牵引车 warm-start
    g_b_tc = ca.DM([L / 2, W / 2, L / 2, W / 2])
    l_tc, n_tc, d_tc = DualMultWS.solve(A, b, nOb, vOb, N, X_g, Y_g, Theta_g,
                                        g_b_tc, L / 2, +1)

    l_g = np.vstack([l_tr, l_tc])          # (2*total_edges, N+1)
    n_g = np.vstack([n_tr, n_tc])          # (8*nOb, N+1)
    sl_g = np.zeros([2*nOb,N+1])        # (2*nOb, N+1)
    return np.concatenate([X_g, Y_g, Theta_g, Psi_g, V_g,
                           l_g.flatten(order='F'),
                           n_g.flatten(order='F'),
                           sl_g.flatten(order='F'),
                           Delta_g, Acc_g, DT_g])


# ==================== 求解 ====================
def solve_parking(nlp, lbx, ubx, lbg, ubg):
    opts = {
        "ipopt.print_level": 5,
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.acceptable_iter": 15,
        "ipopt.constr_viol_tol": 1e-5,
        "ipopt.linear_solver": "mumps",
        "ipopt.mu_strategy": "adaptive",
        "print_time": True,
    }
    solver = ca.nlpsol("trailer_parking", "ipopt", nlp, opts)

    x0 = initial_guess()
    n_var = len(x0)
    n_con = len(lbg)
    print(f"变量数: {n_var}  约束数: {n_con}")
    print("求解中...")

    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    return sol, stats


# ==================== 结果提取 ====================
def extract_results(sol):
    w = np.array(sol["x"]).flatten()
    idx = 0

    def take(n):
        nonlocal idx
        out = w[idx:idx+n]
        idx += n
        return out

    X     = take(N+1); Y     = take(N+1)
    Theta = take(N+1); Psi   = take(N+1); V = take(N+1)
    l_flat = take(2 * total_edges * (N+1)); n_flat = take(8 * nOb * (N+1)); sl_flat = take(2 * nOb * (N+1))
    Delta = take(N);   Acc   = take(N)
    DT    = take(N)

    time = np.zeros(N + 1)
    for k in range(N):
        time[k+1] = time[k] + DT[k]

    l = l_flat.reshape((2 * total_edges, N+1), order='F')
    n = n_flat.reshape((8 * nOb, N+1), order='F')
    sl = sl_flat.reshape((2 * nOb, N+1), order='F')

    return dict(
        X=X, Y=Y, Theta=Theta, Psi=Psi, V=V,
        l=l, n=n, sl=sl,
        Delta=Delta, Acc=Acc, DT=DT,
        time=time, total_time=time[-1]
    )


def print_summary(sol, stats, traj):
    beta_f = traj["Theta"] - traj["Psi"]
    print("\n" + "=" * 55)
    print("半挂牵引车泊车优化结果")
    print("=" * 55)
    print(f"状态:       {stats['return_status']}")
    print(f"总时间:     {traj['total_time']:.3f} s")
    print(f"目标值:     {float(sol['f']):.6f}")
    print(f"迭代:       {stats['iter_count']}")
    print(f"求解耗时:   {stats['t_wall_total']:.3f} s")
    print(f"起点:       ({STATE_0[0]}, {STATE_0[1]})  θ={np.rad2deg(STATE_0[2]):.0f}°  ψ={np.rad2deg(STATE_0[3]):.0f}°")
    print(f"终点:       ({STATE_F[0]}, {STATE_F[1]})  θ={np.rad2deg(STATE_F[2]):.0f}°  ψ={np.rad2deg(STATE_F[3]):.0f}°")
    print(f"末端铰接角: {np.rad2deg(beta_f[-1]):.2f}°")
    print("=" * 55)


# ==================== 可视化 ====================
def plot_trajectory(traj):
    X, Y, Theta, Psi, sl= traj["X"], traj["Y"], traj["Theta"], traj["Psi"], traj["sl"]
    time = traj["time"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # --- Trajectory (xy) ---
    ax = axes[0, 0]
    ax.plot(X, Y, "b-", lw=2, label="Trajectory")
    ax.plot(X[0], Y[0], "go", ms=10, label="Start")
    ax.plot(X[-1], Y[-1], "rs", ms=10, label="Finish")
    trailer_x = X - L_TRAILER * np.cos(Psi)
    trailer_y = Y - L_TRAILER * np.sin(Psi)
    ax.plot(trailer_x, trailer_y, "m-", lw=2)

    square = plt.Rectangle((0.5, 0.5), 1, 1,
                       linewidth=2,
                       edgecolor='blue',
                       facecolor='lightgreen',
                       alpha=0.7)
    ax.add_patch(square)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Trajectory"); ax.legend(); ax.grid(True); ax.axis("equal")

    # --- Position vs Time ---
    ax = axes[0, 1]
    ax.plot(time, X, label="x"); ax.plot(time, Y, label="y")
    ax.set_xlabel("t [s]"); ax.set_ylabel("[m]")
    ax.set_title("Position"); ax.legend(); ax.grid(True)

    # --- sl ---
    ax = axes[0, 2]
    ax.plot(time, sl[0], label="sl_trailer"); ax.plot(time, sl[1], label="sl_tractor")
    ax.set_xlabel("time"); ax.set_ylabel("sl [°]")
    ax.set_title("sl"); ax.grid(True)

    # --- Speed ---
    ax = axes[1, 0]
    ax.plot(time, traj["V"])
    ax.set_xlabel("t [s]"); ax.set_ylabel("v [m/s]")
    ax.set_title("Speed"); ax.grid(True)

    # --- Steering angle ---
    t_mid = 0.5 * (time[:-1] + time[1:])
    ax = axes[1, 1]
    ax.plot(t_mid, np.rad2deg(traj["Delta"]))
    ax.set_xlabel("t [s]"); ax.set_ylabel("δ [°]")
    ax.set_title("Steering Angle"); ax.grid(True)

    # --- Acceleration ---
    ax = axes[1, 2]
    ax.plot(t_mid, traj["Acc"])
    ax.set_xlabel("t [s]"); ax.set_ylabel("a [m/s²]")
    ax.set_title("Acceleration"); ax.grid(True)

    plt.tight_layout()
    plt.savefig("trailer_parking_trajectory.png", dpi=150)
    plt.show()


# ==================== 主函数 ====================
def main():
    print("构建半挂牵引车泊车优化问题...")
    nlp, lbx, ubx, lbg, ubg = build_problem()

    sol, stats = solve_parking(nlp, lbx, ubx, lbg, ubg)

    traj = extract_results(sol)
    print_summary(sol, stats, traj)
    # np.savez('traj2.npz', traj=traj);print("traj saved")
    try:
        plot_trajectory(traj)
    except Exception as e:
        print(f"绘图跳过: {e}")

    try:
        from PyGamePlot2 import visualize
    except ImportError as e:
        print(f"PyGame 动画跳过: {e}")
    else:
        obstacles_list = []
        edge_start = 0
        for nv in vOb:
            obstacles_list.append({
                "A": A[edge_start:edge_start + nv],
                "b": b[edge_start:edge_start + nv],
            })
            edge_start += nv
        try:
            visualize(
                traj,
                obstacles=obstacles_list,
                frame_rate=4,
                tractor_length=L,
                trailer_length=L_TRAILER,
                vehicle_width=W,
            )
        except Exception as e:
            print(f"PyGame 动画失败: {e}")
    return sol, stats, traj


if __name__ == "__main__":
    main()
