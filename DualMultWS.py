import casadi as ca
import numpy as np


def solve(A, b, nOb, vOb, N, X, Y, angle, g_b, offset, offset_sign=-1):
    """Warm-start: 求解一个车身（挂车或牵引车）与障碍物的对偶距离。

    Parameters
    ----------
    angle : ndarray, shape (N+1,)    车身朝向 (Psi 或 Theta)
    g_b : list or DM                车身半长宽 [half_len, half_w, half_len, half_w]
    offset : float                  参考点偏移量
    offset_sign : int               -1 表示向后偏移（挂车），+1 表示向前偏移（牵引车）
    """
    total_edges = sum(vOb)
    l = ca.SX.sym("lambda", total_edges, N + 1)
    n = ca.SX.sym("miu", 4 * nOb, N + 1)
    d = ca.SX.sym("d", nOb, N + 1)

    # ---- 约束 ----
    g, lbg, ubg = [], [], []
    lbx = [0]*((total_edges+4*nOb)*(N+1)) + [-100]*(nOb*(N+1))
    ubx = [ca.inf]*((total_edges+4*nOb)*(N+1)) + [100]*(nOb*(N+1))

    edge_offset = [0]
    for nv in vOb:
        edge_offset.append(edge_offset[-1] + nv)

    for k in range(N + 1):
        for j in range(nOb):
            e_start = edge_offset[j]
            e_end   = edge_offset[j + 1]
            Aj = A[e_start:e_end, :]
            bj = b[e_start:e_end]

            # norm(A'*lambda) == 1
            V_x = ca.dot(Aj[:, 0], l[e_start:e_end, k])
            V_y = ca.dot( Aj[:, 1], l[e_start:e_end, k])
            g.append(V_x**2 + V_y**2);lbg.append(0.0);ubg.append(1.0)

            # G'*mu + R'*A*lambda = 0
            n_j = n[4*j:4*j+4, k]
            x_constrain = n_j[0] - n_j[2] + ca.cos(angle[k])*V_x + ca.sin(angle[k])*V_y
            y_constrain = n_j[1] - n_j[3] - ca.sin(angle[k])*V_x + ca.cos(angle[k])*V_y
            g.append(x_constrain);lbg.append(0.0);ubg.append(0.0)
            g.append(y_constrain);lbg.append(0.0);ubg.append(0.0)

            # -g'*mu + (A*t - b)*lambda > 0
            ref_x = X[k] + offset_sign * ca.cos(angle[k]) * offset
            ref_y = Y[k] + offset_sign * ca.sin(angle[k]) * offset
            distSigned = (-ca.dot(g_b, n_j) +
                         ref_x * V_x + ref_y * V_y -
                         ca.dot(bj, l[e_start:e_end, k]) - d[j, k])
            g.append(distSigned);lbg.append(0.0);ubg.append(0.0)

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
    w = ca.vertcat(l.reshape((-1, 1)), n.reshape((-1, 1)), d.reshape((-1, 1)))
    objective = -ca.sum(d)
    nlp = {"x": w, "f": objective, "g": ca.vertcat(*g)}
    solver = ca.nlpsol("trailer_parking", "ipopt", nlp, opts)
    x0 = np.zeros((1, (total_edges + 4 * nOb + nOb) * (N + 1)))
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    l,n,d = extract_results(sol, N, total_edges, nOb)

    return l,n,d

def extract_results(sol, N, total_edges, nOb):
    w = np.array(sol["x"]).flatten()
    idx = 0

    def take(n):
        nonlocal idx
        out = w[idx:idx+n]
        idx += n
        return out

    l_flat = take(total_edges * (N+1)); n_flat = take(4 * nOb * (N+1)); d = take(nOb * (N+1))
    l = l_flat.reshape((total_edges, N+1), order='F')
    n = n_flat.reshape((4 * nOb, N+1), order='F')
    d = d.reshape((nOb, N+1), order='F')

    return l,n,d

if __name__ == "__main__":
    N = 40
    L = 2.5  # 牵引车轴距 [m]
    W = 1.0  # 轴宽
    L_TRAILER = 7.5  # 铰接点到挂车车轴距离 [m]
    nOb = 1  # 障碍物数量
    vOb = [4, 4]  # 每个障碍物的边数
    A = np.array([[0, 1],
                  [1, 0],
                  [0, -1],
                  [-1, 0],
                  [0, 1],
                  [1, 0],
                  [0, -1],
                  [-1, 0]])
    b = np.array([15.0, 12.0, -9.0, 10.0,
                  8.5, 18.0, 10.0, -12.0
                  ])
    data = np.load('traj.npz', allow_pickle=True)
    traj = data['traj'].item()

    X_g = traj['X']
    Y_g = traj['Y']
    Theta_g = traj['Theta']
    Psi_g = traj['Psi']
    V_g = traj['V']

    Delta_g = traj['Delta']
    Acc_g = traj['Acc']
    DT_g = traj['DT']

    # 挂车 warm-start
    g_b_tr = ca.DM([L_TRAILER / 2, W / 2, L_TRAILER / 2, W / 2])
    l_tr, n_tr, d_tr = solve(A, b, nOb, vOb, N, X_g, Y_g, Psi_g,
                                        g_b_tr, L_TRAILER / 2, -1)
    # 牵引车 warm-start
    g_b_tc = ca.DM([L / 2, W / 2, L / 2, W / 2])
    l_tc, n_tc, d_tc = solve(A, b, nOb, vOb, N, X_g, Y_g, Theta_g,
                                        g_b_tc, L / 2, +1)

    print(d_tr)
    print(d_tc)


