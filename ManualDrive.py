"""Interactive manual drive of semi-trailer truck using PyGame.

Arrow keys:  ← → steer,  ↑ ↓ forward/backward
ESC/Q: quit   R: reset to start
"""

import math
import numpy as np
import pygame

from TrailerParkingOptimization2 import (
    L, W, L_TRAILER, DELTA_BOUNDS, A, b, nOb, vOb, STATE_0, STATE_F,
)

# ==================== 渲染辅助 ====================
COLORS = {
    "bg": (15, 15, 18),
    "tractor": (240, 220, 120),
    "trailer": (255, 165, 100),
    "tractor_path": (86, 180, 255),
    "trailer_path": (255, 120, 220),
    "obstacle": (120, 120, 130),
    "start": (0, 210, 120),
    "finish": (255, 90, 90),
    "text": (220, 220, 220),
}


def _poly_vertices(A, b):
    """Convert half-plane representation A·x ≤ b to CCW polygon vertices."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = len(b)
    verts = []
    for i in range(n):
        j = (i + 1) % n
        det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
        if abs(det) < 1e-12:
            continue
        x = (b[i] * A[j, 1] - b[j] * A[i, 1]) / det
        y = (A[i, 0] * b[j] - A[j, 0] * b[i]) / det
        if np.all(A @ [x, y] <= b + 1e-9):
            verts.append((x, y))
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    verts.sort(key=lambda v: math.atan2(v[1] - cy, v[0] - cx))
    return verts


def _seg_rect(p0, p1, width):
    """Four corners of a rectangle along segment p0→p1 with given width."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        hw = width / 2
        return [(x0 - hw, y0 - hw), (x0 + hw, y0 - hw),
                (x0 + hw, y0 + hw), (x0 - hw, y0 + hw)]
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    hw = width / 2
    return [(x0 + px * hw, y0 + py * hw),
            (x1 + px * hw, y1 + py * hw),
            (x1 - px * hw, y1 - py * hw),
            (x0 - px * hw, y0 - py * hw)]


# ==================== 动力学 ====================
def trailer_ode(x, y, theta, psi, v, delta, a=0.0):
    """半挂牵引车模型状态导数 (numpy version)."""
    beta = theta - psi
    return (
        v * math.cos(theta),
        v * math.sin(theta),
        v * math.tan(delta) / L,
        v * math.sin(beta) / L_TRAILER,
        a,
    )


def euler_step(state, delta, dt, n_sub=10):
    """欧拉积分一步，细分为 n_sub 子步."""
    x, y, theta, psi, v = state
    sub_dt = dt / n_sub
    for _ in range(n_sub):
        dx, dy, dtheta, dpsi, dv = trailer_ode(x, y, theta, psi, v, delta)
        x += dx * sub_dt
        y += dy * sub_dt
        theta += dtheta * sub_dt
        psi += dpsi * sub_dt
        v += dv * sub_dt
    return (x, y, theta, psi, v)


# ==================== 主循环 ====================
def main():
    max_steer = DELTA_BOUNDS[1]       # 35° in radians
    dt_step = 0.2                      # 每步时长 [s]

    state = list(STATE_0)              # [x, y, theta, psi, v]
    t = 0.0

    traj_x, traj_y = [state[0]], [state[1]]
    traj_theta, traj_psi = [state[2]], [state[3]]

    # 障碍物多边形
    obstacles = []
    edge_start = 0
    for nv in vOb:
        obstacles.append({
            "A": A[edge_start:edge_start + nv],
            "b": b[edge_start:edge_start + nv],
        })
        edge_start += nv
    obs_polys = [_poly_vertices(obs["A"], obs["b"]) for obs in obstacles]

    # 世界坐标范围 (起点/终点 + 障碍物)
    xs = [STATE_0[0], STATE_F[0]]
    ys = [STATE_0[1], STATE_F[1]]
    for poly in obs_polys:
        for vx, vy in poly:
            xs.append(vx); ys.append(vy)
    padding = 2.0
    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding

    ww, wh = 900, 700
    scale = min((ww - 80) / (max_x - min_x), (wh - 80) / (max_y - min_y))

    def to_screen(x, y):
        return (int(round(40 + (x - min_x) * scale)),
                int(round(wh - 40 - (y - min_y) * scale)))

    obs_screen = [[to_screen(x, y) for x, y in poly] for poly in obs_polys]
    start_pt = to_screen(STATE_0[0], STATE_0[1])
    finish_pt = to_screen(STATE_F[0], STATE_F[1])

    # PyGame 初始化
    pygame.init()
    screen = pygame.display.set_mode((ww, wh))
    pygame.display.set_caption("Manual Drive")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    running = True
    reached = False

    # 操控状态：左右+上下同时按住才走一步，每次按下组合只走一步
    steer_held = False
    speed_held = False
    steer_value = 0.0
    speed_value = 0.0
    step_consumed = False

    while running:
        dt_frame = clock.tick(60) / 1000.0

        # ---- 事件处理 ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    state = list(STATE_0)
                    t = 0.0
                    traj_x, traj_y = [state[0]], [state[1]]
                    traj_theta, traj_psi = [state[2]], [state[3]]
                    reached = False
                    steer_held = False; speed_held = False
                    steer_value = 0.0; speed_value = 0.0
                    step_consumed = False
                elif event.key == pygame.K_LEFT:
                    steer_held = True; steer_value = max_steer
                elif event.key == pygame.K_RIGHT:
                    steer_held = True; steer_value = -max_steer
                elif event.key == pygame.K_UP:
                    speed_held = True; speed_value = 1.0
                elif event.key == pygame.K_DOWN:
                    speed_held = True; speed_value = -1.0
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    steer_held = False; steer_value = 0.0
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    speed_held = False; speed_value = 0.0

        # 左右+上下同时按住时触发一步，每次按下组合只触发一次
        if steer_held and speed_held and not step_consumed and not reached:
            state[4] = speed_value
            state = list(euler_step(state, steer_value, dt_step))
            t += dt_step
            step_consumed = True
            traj_x.append(state[0]); traj_y.append(state[1])
            traj_theta.append(state[2]); traj_psi.append(state[3])

        if not (steer_held and speed_held):
            step_consumed = False

        # ---- 到达检测 ----
        dist = math.hypot(state[0] - STATE_F[0], state[1] - STATE_F[1])
        if dist < 0.3 and not reached:
            reached = True
            print(f"Arrived!  t = {t:.1f}s  dist = {dist:.3f}m")

        # ---- 渲染 ----
        screen.fill(COLORS["bg"])

        for poly in obs_screen:
            if poly:
                pygame.draw.polygon(screen, COLORS["obstacle"], poly)

        tractor_pts = [to_screen(x, y) for x, y in zip(traj_x, traj_y)]
        trailer_x_all = [tx - L_TRAILER * math.cos(ps) for tx, ps in zip(traj_x, traj_psi)]
        trailer_y_all = [ty - L_TRAILER * math.sin(ps) for ty, ps in zip(traj_y, traj_psi)]
        trailer_pts = [to_screen(x, y) for x, y in zip(trailer_x_all, trailer_y_all)]

        if len(tractor_pts) > 1:
            pygame.draw.lines(screen, COLORS["tractor_path"], False, tractor_pts, 3)
            pygame.draw.lines(screen, COLORS["trailer_path"], False, trailer_pts, 3)

        pygame.draw.circle(screen, COLORS["start"], start_pt, 8)
        pygame.draw.circle(screen, COLORS["finish"], finish_pt, 8, width=2)

        # 当前车辆
        x, y, theta, psi, v = state
        fw = (x + L * math.cos(theta), y + L * math.sin(theta))
        tw = _seg_rect((x, y), fw, W)
        pygame.draw.polygon(screen, COLORS["tractor"], [to_screen(*p) for p in tw])

        tr_axle = (x - L_TRAILER * math.cos(psi), y - L_TRAILER * math.sin(psi))
        trw = _seg_rect((x, y), tr_axle, W)
        pygame.draw.polygon(screen, COLORS["trailer"], [to_screen(*p) for p in trw])

        hitch_pt = to_screen(x, y)
        trailer_pt = to_screen(*tr_axle)
        pygame.draw.circle(screen, COLORS["tractor"], hitch_pt, 4)
        pygame.draw.circle(screen, COLORS["trailer"], trailer_pt, 4)

        # HUD
        def hud(text, row):
            screen.blit(font.render(text, True, COLORS["text"]), (20, 15 + row * 22))

        hud(f"t = {t:6.2f}s   steps: {len(traj_x)}", 0)
        hud(f"x = {x:+.2f}  y = {y:+.2f}", 1)
        hud(f"theta = {math.degrees(theta):+.1f} deg   psi = {math.degrees(psi):+.1f} deg", 2)
        hud(f"v = {v:+.2f} m/s   delta = {math.degrees(steer_value):+.0f} deg", 3)
        hud(f"dist to goal: {dist:.2f}m", 4)

        if reached:
            hud("ARRIVED!", 6)
        hud("L/R + U/D to step  R: reset  ESC/Q: quit", 7)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
