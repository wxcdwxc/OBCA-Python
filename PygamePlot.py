"""Pygame-based visualization for trailer parking trajectory."""
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:  # Optional dependency
    import pygame
except ImportError as exc:  # pragma: no cover - visualization only
    pygame = None  # type: ignore
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None


Color = Tuple[int, int, int]
Point = Tuple[int, int]
Trajectory = Dict[str, np.ndarray]


DEFAULT_COLORS = {
    "background": (15, 15, 18),
    "grid": (55, 55, 65),
    "tractor_path": (86, 180, 255),
    "trailer_path": (255, 120, 220),
    "tractor_body": (240, 220, 120),
    "trailer_body": (255, 165, 100),
    "text": (220, 220, 220),
    "start": (0, 210, 120),
    "finish": (255, 90, 90),
    "obstacle": (120, 120, 130),
    "safe": (220, 120, 120),
}


def _require_pygame() -> None:
    if pygame is None:  # pragma: no cover - visualization only
        raise ImportError(
            "pygame 未安装，无法显示动画。请运行 `pip install pygame` 后重试。"
        ) from _PYGAME_IMPORT_ERROR


def _normalize_obstacle(obstacle: Any) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """Normalize obstacle input to a canonical shape.

    Supported:
    - (ox, oy, r) -> circle
    - (x, y, w, h) -> axis-aligned rectangle in world coords
    - {"A": array_like(4,2), "b": array_like(4,)} for an axis-aligned box
      with rows [0,1],[1,0],[0,-1],[-1,0] (same as TrailerParkingOptimization.py).
    """
    if obstacle is None:
        return None

    if isinstance(obstacle, dict) and "A" in obstacle and "b" in obstacle:
        A = np.asarray(obstacle["A"], dtype=float)
        b = np.asarray(obstacle["b"], dtype=float).reshape(-1)
        if A.shape == (4, 2) and b.shape == (4,):
            # y <= b0, x <= b1, -y <= b2 => y >= -b2, -x <= b3 => x >= -b3
            x_min = float(-b[3])
            x_max = float(b[1])
            y_min = float(-b[2])
            y_max = float(b[0])
            return ("rect", (x_min, y_min, x_max - x_min, y_max - y_min))

    if isinstance(obstacle, (tuple, list)):
        if len(obstacle) == 3:
            ox, oy, r = obstacle
            return ("circle", (float(ox), float(oy), float(r)))
        if len(obstacle) == 4:
            x, y, w, h = obstacle
            return ("rect", (float(x), float(y), float(w), float(h)))

    return None


def _compute_bounds(
    tractor_pts: Iterable[Tuple[float, float]],
    trailer_pts: Iterable[Tuple[float, float]],
    obstacle: Any,
    padding: float,
) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for x, y in tractor_pts:
        xs.append(x)
        ys.append(y)
    for x, y in trailer_pts:
        xs.append(x)
        ys.append(y)

    norm_obstacle = _normalize_obstacle(obstacle)
    if norm_obstacle is not None:
        kind, params = norm_obstacle
        if kind == "circle":
            ox, oy, r = params
            xs.extend([ox - r, ox + r])
            ys.extend([oy - r, oy + r])
        elif kind == "rect":
            x, y, w, h = params
            xs.extend([x, x + w])
            ys.extend([y, y + h])

    if not xs:
        xs = [0.0]
        ys = [0.0]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if math.isclose(min_x, max_x):
        min_x -= 0.5
        max_x += 0.5
    if math.isclose(min_y, max_y):
        min_y -= 0.5
        max_y += 0.5

    return min_x - padding, max_x + padding, min_y - padding, max_y + padding


def _segment_rectangle(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Rectangle corners for a segment p0->p1 with given width (world units)."""
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        # Degenerate: return a tiny square around p0.
        half = width / 2.0
        return ((x0 - half, y0 - half), (x0 + half, y0 - half), (x0 + half, y0 + half), (x0 - half, y0 + half))
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    half = width / 2.0
    return (
        (x0 + px * half, y0 + py * half),
        (x1 + px * half, y1 + py * half),
        (x1 - px * half, y1 - py * half),
        (x0 - px * half, y0 - py * half),
    )


def visualize(
    traj: Trajectory,
    frame_rate: int = 1,
    window_size: Tuple[int, int] = (900, 700),
    padding: float = 0.6,
    obstacle: Any = None,
    safe_radius: Optional[float] = None,
    tractor_length: float = 2.5,
    trailer_length: float = 7.5,
    vehicle_width: float = 1.0,
) -> None:
    """Display Pygame animation for the solved trajectory."""

    _require_pygame()
    X = np.asarray(traj["X"], dtype=float)
    Y = np.asarray(traj["Y"], dtype=float)
    Theta = np.asarray(traj["Theta"], dtype=float)
    Psi = np.asarray(traj["Psi"], dtype=float)
    time = np.asarray(traj.get("time", np.arange(len(X))), dtype=float)

    trailer_x = X - trailer_length * np.cos(Psi)
    trailer_y = Y - trailer_length * np.sin(Psi)
    tractor_pts_world = list(zip(X, Y))
    trailer_pts_world = list(zip(trailer_x, trailer_y))

    min_x, max_x, min_y, max_y = _compute_bounds(
        tractor_pts_world, trailer_pts_world, obstacle, padding
    )

    window_w, window_h = window_size
    usable_w = window_w - 80
    usable_h = window_h - 80
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min(usable_w / span_x, usable_h / span_y)

    def to_screen(x: float, y: float) -> Point:
        sx = 40 + (x - min_x) * scale
        sy = window_h - (40 + (y - min_y) * scale)
        return int(round(sx)), int(round(sy))

    tractor_pts_screen = [to_screen(x, y) for x, y in tractor_pts_world]
    trailer_pts_screen = [to_screen(x, y) for x, y in trailer_pts_world]

    pygame.init()  # pragma: no cover - visualization only
    try:
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Trailer Parking Animation")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 16)

        idx = 0
        running = True
        num_frames = len(X)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        idx = 0

            screen.fill(DEFAULT_COLORS["background"])

            norm_obstacle = _normalize_obstacle(obstacle)
            if norm_obstacle is not None:
                kind, params = norm_obstacle
                if kind == "circle":
                    ox, oy, radius = params
                    center = to_screen(ox, oy)
                    pygame.draw.circle(
                        screen,
                        DEFAULT_COLORS["obstacle"],
                        center,
                        max(1, int(radius * scale)),
                    )
                    if safe_radius is not None:
                        pygame.draw.circle(
                            screen,
                            DEFAULT_COLORS["safe"],
                            center,
                            max(1, int(safe_radius * scale)),
                            width=2,
                        )
                elif kind == "rect":
                    x, y, w, h = params
                    p0 = to_screen(x, y)
                    p1 = to_screen(x + w, y)
                    p2 = to_screen(x + w, y + h)
                    p3 = to_screen(x, y + h)
                    pygame.draw.polygon(
                        screen,
                        DEFAULT_COLORS["obstacle"],
                        [p0, p1, p2, p3],
                    )
                    if safe_radius is not None:
                        x2 = x - safe_radius
                        y2 = y - safe_radius
                        w2 = w + 2.0 * safe_radius
                        h2 = h + 2.0 * safe_radius
                        q0 = to_screen(x2, y2)
                        q1 = to_screen(x2 + w2, y2)
                        q2 = to_screen(x2 + w2, y2 + h2)
                        q3 = to_screen(x2, y2 + h2)
                        pygame.draw.polygon(
                            screen,
                            DEFAULT_COLORS["safe"],
                            [q0, q1, q2, q3],
                            width=2,
                        )

            pygame.draw.circle(screen, DEFAULT_COLORS["start"], tractor_pts_screen[0], 8)
            pygame.draw.circle(
                screen, DEFAULT_COLORS["finish"], tractor_pts_screen[-1], 8, width=2
            )

            if idx > 0:
                pygame.draw.lines(
                    screen,
                    DEFAULT_COLORS["tractor_path"],
                    False,
                    tractor_pts_screen[: idx + 1],
                    3,
                )
                pygame.draw.lines(
                    screen,
                    DEFAULT_COLORS["trailer_path"],
                    False,
                    trailer_pts_screen[: idx + 1],
                    3,
                )

            hitch = tractor_pts_screen[idx]
            front_world = (
                X[idx] + tractor_length * math.cos(Theta[idx]),
                Y[idx] + tractor_length * math.sin(Theta[idx]),
            )
            front = to_screen(*front_world)
            trailer_axle = trailer_pts_screen[idx]

            tractor_rect_world = _segment_rectangle((X[idx], Y[idx]), front_world, vehicle_width)
            tractor_rect_screen = [to_screen(x, y) for x, y in tractor_rect_world]
            pygame.draw.polygon(screen, DEFAULT_COLORS["tractor_body"], tractor_rect_screen)

            trailer_rect_world = _segment_rectangle((X[idx], Y[idx]), (trailer_x[idx], trailer_y[idx]), vehicle_width)
            trailer_rect_screen = [to_screen(x, y) for x, y in trailer_rect_world]
            pygame.draw.polygon(screen, DEFAULT_COLORS["trailer_body"], trailer_rect_screen)

            # Keep articulation/axle markers for clarity.
            pygame.draw.circle(screen, DEFAULT_COLORS["tractor_body"], hitch, 4)
            pygame.draw.circle(screen, DEFAULT_COLORS["trailer_body"], trailer_axle, 4)

            t_val = time[idx] if idx < len(time) else time[-1]
            info = f"t = {t_val:5.2f} s    frame {idx + 1}/{num_frames}"
            text_surface = font.render(info, True, DEFAULT_COLORS["text"])
            screen.blit(text_surface, (20, 15))
            help_surface = font.render("ESC/Q: 退出   R: 重播", True, DEFAULT_COLORS["text"])
            screen.blit(help_surface, (20, 40))

            pygame.display.flip()
            clock.tick(frame_rate)
            if idx < num_frames - 1:
                idx += 1

    finally:  # pragma: no cover - visualization only
        pygame.quit()
