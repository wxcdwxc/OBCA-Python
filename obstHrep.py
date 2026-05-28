"""
Compute H-representation (A, b) for convex obstacles from vertex coordinates.

Based on the OBCA algorithm (Zhang, Liniger & Borrelli, 2017),
ported from the Julia reference implementation.

For each edge of a clockwise polygon, computes A_i, b_i such that
the obstacle is { x | A_i @ x <= b_i for all i }.
"""

import numpy as np


def compute_A_b(vertices):
    """
    Compute half-plane representation for a single convex obstacle.

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)
        Vertices in CLOCKWISE order.  Do NOT repeat the first vertex.

    Returns
    -------
    A : ndarray, shape (n, 2)
    b : ndarray, shape (n,)

    Obstacle = { x | A @ x <= b }.
    """
    n = len(vertices)
    A = np.zeros((n, 2))
    b = np.zeros(n)

    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]

        if v1[0] == v2[0]:          # vertical edge
            if v2[1] < v1[1]:       # going down (clockwise)
                A[i] = [1.0, 0.0];   b[i] = v1[0]
            else:                    # going up
                A[i] = [-1.0, 0.0];  b[i] = -v1[0]

        elif v1[1] == v2[1]:        # horizontal edge
            if v1[0] < v2[0]:       # going right (clockwise)
                A[i] = [0.0, 1.0];   b[i] = v1[1]
            else:                    # going left
                A[i] = [0.0, -1.0];  b[i] = -v1[1]

        else:                       # slanted edge
            # line through v1, v2:  y = a * x + b_val
            M = np.array([[v1[0], 1.0],
                          [v2[0], 1.0]])
            rhs = np.array([v1[1], v2[1]])
            a_coef, b_val = np.linalg.solve(M, rhs)

            if v1[0] < v2[0]:       # moving right → interior is below the line
                A[i] = [-a_coef, 1.0];  b[i] = b_val
            else:                    # moving left  → interior is above the line
                A[i] = [a_coef, -1.0];  b[i] = -b_val

    return A, b


def obstHrep(obstacles):
    """
    Compute H-representation for multiple obstacles.

    Parameters
    ----------
    obstacles : list of ndarray, each shape (n_i, 2)
        Clockwise vertices per obstacle.  First vertex not repeated.

    Returns
    -------
    A_all : ndarray, shape (total_edges, 2)
        Stacked A matrices.
    b_all : ndarray, shape (total_edges,)
        Stacked b vectors.
    n_edges : list of int
        Number of edges per obstacle, useful for splitting later.
    """
    A_list, b_list = [], []
    n_edges = []

    for verts in obstacles:
        A_i, b_i = compute_A_b(verts)
        A_list.append(A_i)
        b_list.append(b_i)
        n_edges.append(len(verts))

    A_all = np.vstack(A_list) if A_list else np.empty((0, 2))
    b_all = np.concatenate(b_list) if b_list else np.empty(0)

    return A_all, b_all, n_edges


# ==================== 测试 ====================
if __name__ == "__main__":
    # 一个正方形：中心(2,2), 边长1 → x∈[1.5,2.5], y∈[1.5,2.5]
    # 顺时针 (CW) 顶点
    square = np.array([
        [1.0, 1.5],
        [1.5, 5.0],
        [3.0, 2.5],
        [2.0, 1.0],
    ])
    A, b = compute_A_b(square)
    print("=== 正方形 (中心2,2 边长1) ===")
    print(f"A =\n{A}")
    print(f"b = {b}")
    # 期望:
    #   边1: x=1.5, y∈[1.5,2.5], 向上 → A=[-1,0], b=-1.5  ( -x ≤ -1.5 → x ≥ 1.5 )
    #   边2: y=2.5, x∈[1.5,2.5], 向右 → A=[0,1],  b=2.5   ( y ≤ 2.5 )
    #   边3: x=2.5, y∈[2.5,1.5], 向下 → A=[1,0],  b=2.5   ( x ≤ 2.5 )
    #   边4: y=1.5, x∈[2.5,1.5], 向左 → A=[0,-1], b=-1.5  ( -y ≤ -1.5 → y ≥ 1.5 )

    # 多个障碍物 (CW)
    rect = np.array([
        [6.5, 0.0],
        [6.5, 1.0],
        [7.5, 1.0],
        [7.5, 0.0],
    ])
    A_all, b_all, n_edges = obstHrep([square, rect])
    print(f"\n=== 两个障碍物 ===")
    print(f"A_all =\n{A_all}")
    print(f"b_all = {b_all}")
    print(f"每个障碍物边数: {n_edges}")
    print(f"总边数: {len(A_all)}")
