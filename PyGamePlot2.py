"""Minimal trajectory + polygon-obstacle visualization."""
import math

import numpy as np
import pygame

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


def visualize(traj, obstacles=None, frame_rate=1, window_size=(900, 700),
              padding=0.6, tractor_length=2.5, trailer_length=7.5,
              vehicle_width=1.0):
    X = np.asarray(traj["X"], dtype=float)
    Y = np.asarray(traj["Y"], dtype=float)
    Theta = np.asarray(traj["Theta"], dtype=float)
    Psi = np.asarray(traj["Psi"], dtype=float)
    time = np.asarray(traj.get("time", np.arange(len(X))), dtype=float)

    trailer_x = X - trailer_length * np.cos(Psi)
    trailer_y = Y - trailer_length * np.sin(Psi)

    # Collect world bounds from trajectory and obstacles
    xs = list(X) + list(trailer_x)
    ys = list(Y) + list(trailer_y)
    obs_polys = []
    if obstacles:
        for obs in obstacles:
            verts = _poly_vertices(obs["A"], obs["b"])
            obs_polys.append(verts)
            for v in verts:
                xs.append(v[0])
                ys.append(v[1])

    min_x = min(xs) - padding
    max_x = max(xs) + padding
    min_y = min(ys) - padding
    max_y = max(ys) + padding
    if max_x - min_x < 1e-9:
        min_x -= 0.5
        max_x += 0.5
    if max_y - min_y < 1e-9:
        min_y -= 0.5
        max_y += 0.5

    ww, wh = window_size
    scale = min((ww - 80) / (max_x - min_x), (wh - 80) / (max_y - min_y))

    def to_screen(x, y):
        return (int(round(40 + (x - min_x) * scale)),
                int(round(wh - 40 - (y - min_y) * scale)))

    tractor_pts = [to_screen(x, y) for x, y in zip(X, Y)]
    obs_screen = [[to_screen(x, y) for x, y in poly] for poly in obs_polys]

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Trajectory Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    idx = 0
    running = True
    N = len(X)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    idx = 0

        screen.fill(COLORS["bg"])

        for poly in obs_screen:
            if poly:
                pygame.draw.polygon(screen, COLORS["obstacle"], poly)

        if idx > 0:
            pygame.draw.lines(
                screen, COLORS["tractor_path"], False, tractor_pts[:idx + 1], 3
            )

        pygame.draw.circle(screen, COLORS["start"], tractor_pts[0], 8)
        pygame.draw.circle(screen, COLORS["finish"], tractor_pts[-1], 8, width=2)

        hitch = tractor_pts[idx]
        fw = (X[idx] + tractor_length * math.cos(Theta[idx]),
              Y[idx] + tractor_length * math.sin(Theta[idx]))
        front = to_screen(*fw)

        tw = _seg_rect((X[idx], Y[idx]), fw, vehicle_width)
        pygame.draw.polygon(
            screen, COLORS["tractor"],
            [to_screen(x, y) for x, y in tw]
        )
        trw = _seg_rect((X[idx], Y[idx]),
                        (trailer_x[idx], trailer_y[idx]), vehicle_width)
        pygame.draw.polygon(
            screen, COLORS["trailer"],
            [to_screen(x, y) for x, y in trw]
        )

        pygame.draw.circle(screen, COLORS["tractor"], hitch, 4)

        t_val = time[idx]
        info = f"t = {t_val:5.2f}s  {idx + 1}/{N}"
        screen.blit(font.render(info, True, COLORS["text"]), (20, 15))
        screen.blit(
            font.render("ESC/Q: quit  R: reset", True, COLORS["text"]), (20, 40)
        )

        pygame.display.flip()
        clock.tick(frame_rate)
        if idx < N - 1:
            idx += 1

    pygame.quit()
