"""Hybrid A* path planner (Python port of hybrid_a_star.jl).

This module mirrors the Julia implementation found in D:/new/A_star/hybrid_a_star.jl
and preserves the structure, constants, and helper routines so it can be paired with
future Python ports of `reeds_shepp`, `collision_check`, and `a_star`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # SciPy provides an efficient KD-tree implementation
    from scipy.spatial import cKDTree as _KDTreeBase  # type: ignore
except Exception:  # pragma: no cover - SciPy may not be installed
    _KDTreeBase = None

try:
    import reeds_shepp  # type: ignore
except Exception:  # pragma: no cover - stub fallback used until real module exists
    class _StubReedsSheppPath:
        def __init__(self):
            self.lengths: List[float] = []
            self.ctypes: List[str] = []
            self.x: List[float] = []
            self.y: List[float] = []
            self.yaw: List[float] = []

    class _StubReedsSheppModule:
        Path = _StubReedsSheppPath

        @staticmethod
        def calc_shortest_path(*_args, **_kwargs):
            raise NotImplementedError(
                "reeds_shepp.calc_shortest_path is not implemented in this stub."
            )

        @staticmethod
        def calc_shortest_path_length(*_args, **_kwargs):
            raise NotImplementedError(
                "reeds_shepp.calc_shortest_path_length is not implemented in this stub."
            )

    reeds_shepp = _StubReedsSheppModule()  # type: ignore

try:
    import collision_check  # type: ignore
except Exception:  # pragma: no cover - stub fallback used until real module exists
    class _StubCollisionCheck:
        @staticmethod
        def check_collision(*_args, **_kwargs):
            raise NotImplementedError(
                "collision_check.check_collision is not implemented in this stub."
            )

    collision_check = _StubCollisionCheck()  # type: ignore

try:
    import a_star  # type: ignore
except Exception:  # pragma: no cover - stub fallback used until real module exists
    class _StubAStar:
        @staticmethod
        def calc_dist_policy(*_args, **_kwargs):
            raise NotImplementedError(
                "a_star.calc_dist_policy is not implemented in this stub."
            )

    a_star = _StubAStar()  # type: ignore


class _SimpleKDTree:
    """Brute-force KD-tree fallback when SciPy is unavailable."""

    def __init__(self, points: np.ndarray):
        self._points = np.asarray(points, dtype=float)

    def query(self, point: Sequence[float], k: int = 1):
        if self._points.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        point_arr = np.asarray(point, dtype=float)
        dists = np.linalg.norm(self._points - point_arr, axis=1)
        idx = np.argsort(dists)[:k]
        return idx, dists[idx]


def _build_kdtree(points: np.ndarray):
    if _KDTreeBase is not None and points.size > 0:
        return _KDTreeBase(points)
    return _SimpleKDTree(points)


VEHICLE_RADIUS = 1.0  # [m]
BUBBLE_DIST = 1.7  # [m]
OB_MAP_RESOLUTION = 0.1  # [m]
YAW_GRID_RESOLUTION = math.radians(5.0)
N_STEER = 5.0
XY_GRID_RESOLUTION = 0.3  # [m]
MOTION_RESOLUTION = 0.1  # [m]
USE_HOLONOMIC_WITH_OBSTACLE_HEURISTIC = True
USE_NONHOLONOMIC_WITHOUT_OBSTACLE_HEURISTIC = False
SB_COST = 10.0
BACK_COST = 0.0
STEER_CHANGE_COST = 10.0
STEER_COST = 0.0
H_COST = 1.0
WB = 2.7  # [m]
MAX_STEER = 0.6  # [rad]


@dataclass
class Node:
    xind: int
    yind: int
    yawind: int
    direction: bool
    xs: List[float]
    ys: List[float]
    yaws: List[float]
    steer: float
    cost: float
    parent_index: int


@dataclass
class Config:
    minx: int
    miny: int
    minyaw: int
    maxx: int
    maxy: int
    maxyaw: int
    xw: int
    yw: int
    yaww: int
    xyreso: float
    yawreso: float
    obminx: int
    obminy: int
    obmaxx: int
    obmaxy: int
    obxw: int
    obyw: int
    obreso: float


def pi_2_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def calc_euclid_dist(dx: float, dy: float, dyaw: float) -> float:
    if dyaw >= math.pi:
        dyaw -= math.pi
    elif dyaw <= -math.pi:
        dyaw += math.pi
    return math.sqrt(dx * dx + dy * dy + dyaw * dyaw)


def calc_config(
    ox: Sequence[float],
    oy: Sequence[float],
    xyreso: float,
    yawreso: float,
    obreso: float,
) -> Config:
    minx = int(round(min(ox) / xyreso))
    miny = int(round(min(oy) / xyreso))
    maxx = int(round(max(ox) / xyreso))
    maxy = int(round(max(oy) / xyreso))
    obminx = int(round(min(ox) / obreso))
    obminy = int(round(min(oy) / obreso))
    obmaxx = int(round(max(ox) / obreso))
    obmaxy = int(round(max(oy) / obreso))
    xw = int(round(maxx - minx))
    yw = int(round(maxy - miny))
    obxw = int(round(obmaxx - obminx))
    obyw = int(round(obmaxy - obminy))
    minyaw = int(round(-math.pi / yawreso)) - 1
    maxyaw = int(round(math.pi / yawreso))
    yaww = int(round(maxyaw - minyaw))
    return Config(
        minx,
        miny,
        minyaw,
        maxx,
        maxy,
        maxyaw,
        xw,
        yw,
        yaww,
        xyreso,
        yawreso,
        obminx,
        obminy,
        obmaxx,
        obmaxy,
        obxw,
        obyw,
        obreso,
    )


def calc_obstacle_map(
    ox: Sequence[float], oy: Sequence[float], config: Config
) -> Tuple[np.ndarray, object]:
    scale_x = np.asarray(ox, dtype=float) / config.obreso
    scale_y = np.asarray(oy, dtype=float) / config.obreso
    points = np.column_stack((scale_x, scale_y)) if scale_x.size else np.empty((0, 2))
    obmap = np.zeros((config.obxw, config.obyw), dtype=bool)
    gkdtree = _build_kdtree(points)
    for ix in range(config.obxw):
        x = ix + config.obminx
        for iy in range(config.obyw):
            y = iy + config.obminy
            _, dists = gkdtree.query([x, y], k=1)
            if dists.size and dists[0] <= VEHICLE_RADIUS / config.obreso:
                obmap[ix, iy] = True
    return obmap, gkdtree


def calc_holonomic_with_obstacle_heuristic(
    goal: Node, ox: Sequence[float], oy: Sequence[float], xyreso: float
) -> Optional[np.ndarray]:
    try:
        return a_star.calc_dist_policy(
            goal.xs[-1], goal.ys[-1], ox, oy, xyreso, VEHICLE_RADIUS
        )
    except NotImplementedError:
        return None


def calc_nonholonomic_without_obstacle_heuristic(
    goal: Node, config: Config
) -> Optional[np.ndarray]:
    h_rs = np.zeros((config.xw, config.yw, config.yaww), dtype=float)
    max_curvature = math.tan(MAX_STEER) / WB
    for ix in range(config.xw):
        for iy in range(config.yw):
            for iyaw in range(config.yaww):
                sx = (ix + config.minx) * config.xyreso
                sy = (iy + config.miny) * config.xyreso
                syaw = pi_2_pi((iyaw + config.minyaw) * config.yawreso)
                try:
                    length = reeds_shepp.calc_shortest_path_length(
                        sx,
                        sy,
                        syaw,
                        goal.xs[-1],
                        goal.ys[-1],
                        goal.yaws[-1],
                        max_curvature,
                        step_size=MOTION_RESOLUTION,
                    )
                except NotImplementedError:
                    return None
                h_rs[ix, iy, iyaw] = length
    return h_rs


def calc_index(node: Node, config: Config) -> int:
    ind = (
        (node.yawind - config.minyaw) * config.xw * config.yw
        + (node.yind - config.miny) * config.xw
        + (node.xind - config.minx)
    )
    if ind <= 0:
        raise ValueError(f"calc_index produced non-positive index: {ind}")
    return ind


def is_same_grid(node1: Node, node2: Node) -> bool:
    return (
        node1.xind == node2.xind
        and node1.yind == node2.yind
        and node1.yawind == node2.yawind
    )


def calc_motion_inputs() -> Tuple[List[float], List[float]]:
    step = MAX_STEER / N_STEER
    up = [i for i in np.arange(step, MAX_STEER + 1e-6, step)]
    u = [0.0] + up + [-i for i in up]
    d = [1.0 for _ in range(len(u))] + [-1.0 for _ in range(len(u))]
    u = u + u
    return u, d


def calc_next_node(
    current: Node,
    current_id: int,
    steer_input: float,
    direction_multiplier: float,
    config: Config,
    _gkdtree: object,
) -> Node:
    arc_l = XY_GRID_RESOLUTION
    nlist = int(round(arc_l / MOTION_RESOLUTION)) + 1
    xlist = [0.0] * nlist
    ylist = [0.0] * nlist
    yawlist = [0.0] * nlist
    xlist[0] = current.xs[-1] + direction_multiplier * MOTION_RESOLUTION * math.cos(
        current.yaws[-1]
    )
    ylist[0] = current.ys[-1] + direction_multiplier * MOTION_RESOLUTION * math.sin(
        current.yaws[-1]
    )
    yawlist[0] = pi_2_pi(
        current.yaws[-1]
        + direction_multiplier * MOTION_RESOLUTION / WB * math.tan(steer_input)
    )
    for i in range(nlist - 1):
        xlist[i + 1] = xlist[i] + direction_multiplier * MOTION_RESOLUTION * math.cos(
            yawlist[i]
        )
        ylist[i + 1] = ylist[i] + direction_multiplier * MOTION_RESOLUTION * math.sin(
            yawlist[i]
        )
        yawlist[i + 1] = pi_2_pi(
            yawlist[i]
            + direction_multiplier * MOTION_RESOLUTION / WB * math.tan(steer_input)
        )
    xind = int(round(xlist[-1] / config.xyreso))
    yind = int(round(ylist[-1] / config.xyreso))
    yawind = int(round(yawlist[-1] / config.yawreso))
    added_cost = 0.0
    if direction_multiplier > 0:
        direction = True
        added_cost += abs(arc_l)
    else:
        direction = False
        added_cost += abs(arc_l) * BACK_COST
    if direction != current.direction:
        added_cost += SB_COST
    added_cost += STEER_COST * abs(steer_input)
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer_input)
    cost = current.cost + added_cost
    return Node(
        xind,
        yind,
        yawind,
        direction,
        xlist,
        ylist,
        yawlist,
        steer_input,
        cost,
        current_id,
    )


def verify_index(
    node: Node,
    _obmap: np.ndarray,
    config: Config,
    kdtree: object,
    ox: Sequence[float],
    oy: Sequence[float],
) -> bool:
    if (node.xind - config.minx) >= config.xw or (node.xind - config.minx) <= 0:
        return False
    if (node.yind - config.miny) >= config.yw or (node.yind - config.miny) <= 0:
        return False
    try:
        if not collision_check.check_collision(node.xs, node.ys, node.yaws, kdtree, ox, oy):
            return False
    except NotImplementedError:
        pass
    return True


def calc_rs_path_cost(rspath: "reeds_shepp.Path") -> float:  # type: ignore[name-defined]
    cost = 0.0
    for length in rspath.lengths:
        if length >= 0.0:
            cost += length
        else:
            cost += abs(length) * BACK_COST
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += SB_COST
    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += STEER_COST * abs(MAX_STEER)
    steer_profile = [0.0] * len(rspath.ctypes)
    for idx, ctype in enumerate(rspath.ctypes):
        if ctype == "R":
            steer_profile[idx] = -MAX_STEER
        elif ctype == "L":
            steer_profile[idx] = MAX_STEER
    for i in range(len(steer_profile) - 1):
        cost += STEER_CHANGE_COST * abs(steer_profile[i + 1] - steer_profile[i])
    return cost


def analytic_expansion(
    node: Node,
    goal: Node,
    _obmap: np.ndarray,
    _config: Config,
    kdtree: object,
    ox: Sequence[float],
    oy: Sequence[float],
):
    sx = node.xs[-1]
    sy = node.ys[-1]
    syaw = node.yaws[-1]
    max_curvature = math.tan(MAX_STEER) / WB
    try:
        path = reeds_shepp.calc_shortest_path(
            sx,
            sy,
            syaw,
            goal.xs[-1],
            goal.ys[-1],
            goal.yaws[-1],
            max_curvature,
            step_size=MOTION_RESOLUTION,
        )
    except NotImplementedError:
        return None
    if path is None:
        return None
    try:
        if not collision_check.check_collision(path.x, path.y, path.yaw, kdtree, ox, oy):
            return None
    except NotImplementedError:
        pass
    return path


def update_node_with_analytic_expansion(
    current: Node,
    goal: Node,
    obmap: np.ndarray,
    config: Config,
    kdtree: object,
    ox: Sequence[float],
    oy: Sequence[float],
) -> Tuple[bool, Node]:
    path = analytic_expansion(current, goal, obmap, config, kdtree, ox, oy)
    if path is not None:
        current.xs.extend(path.x[1:-1])
        current.ys.extend(path.y[1:-1])
        current.yaws.extend(path.yaw[1:-1])
        current.cost += calc_rs_path_cost(path)
        return True, current
    return False, current


def get_final_path(
    closed: Dict[int, Node], goal: Node, start: Node, config: Config
) -> Tuple[List[float], List[float], List[float]]:
    rx = list(goal.xs)
    ry = list(goal.ys)
    ryaw = list(goal.yaws)
    nid = calc_index(goal, config)
    while True:
        node = closed[nid]
        rx.extend(reversed(node.xs))
        ry.extend(reversed(node.ys))
        ryaw.extend(reversed(node.yaws))
        nid = node.parent_index
        if is_same_grid(node, start):
            break
    rx.reverse()
    ry.reverse()
    ryaw.reverse()
    return rx, ry, ryaw


def calc_cost(
    node: Node,
    h_rs: Optional[np.ndarray],
    h_dp: Optional[np.ndarray],
    goal: Node,
    config: Config,
) -> float:
    has_rs = h_rs is not None and h_rs.size > 0  # type: ignore[arg-type]
    has_dp = h_dp is not None and h_dp.size > 0  # type: ignore[arg-type]
    ix = node.xind - config.minx
    iy = node.yind - config.miny
    iyaw = node.yawind - config.minyaw
    if has_rs and has_dp:
        c_dp = h_dp[ix, iy]
        c_rs = h_rs[ix, iy, iyaw]
        return node.cost + H_COST * max(c_dp, c_rs)
    if has_dp:
        return node.cost + H_COST * h_dp[ix, iy]
    if has_rs:
        return node.cost + H_COST * h_rs[ix, iy, iyaw]
    return node.cost + H_COST * calc_euclid_dist(
        node.xs[-1] - goal.xs[-1],
        node.ys[-1] - goal.ys[-1],
        node.yaws[-1] - goal.yaws[-1],
    )


def calc_hybrid_astar_path(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    ox: Sequence[float],
    oy: Sequence[float],
    xyreso: float,
    yawreso: float,
    obreso: float,
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    syaw = pi_2_pi(syaw)
    gyaw = pi_2_pi(gyaw)
    config = calc_config(ox, oy, xyreso, yawreso, obreso)
    obs_points = np.column_stack((ox, oy)) if ox else np.empty((0, 2))
    kdtree = _build_kdtree(obs_points)
    obmap, gkdtree = calc_obstacle_map(ox, oy, config)
    nstart = Node(
        int(round(sx / xyreso)),
        int(round(sy / xyreso)),
        int(round(syaw / yawreso)),
        True,
        [sx],
        [sy],
        [syaw],
        0.0,
        0.0,
        -1,
    )
    ngoal = Node(
        int(round(gx / xyreso)),
        int(round(gy / xyreso)),
        int(round(gyaw / yawreso)),
        True,
        [gx],
        [gy],
        [gyaw],
        0.0,
        0.0,
        -1,
    )
    h_dp = (
        calc_holonomic_with_obstacle_heuristic(ngoal, ox, oy, xyreso)
        if USE_HOLONOMIC_WITH_OBSTACLE_HEURISTIC
        else None
    )
    h_rs = (
        calc_nonholonomic_without_obstacle_heuristic(ngoal, config)
        if USE_NONHOLONOMIC_WITHOUT_OBSTACLE_HEURISTIC
        else None
    )
    open_set: Dict[int, Node] = {}
    closed_set: Dict[int, Node] = {}
    start_id = calc_index(nstart, config)
    open_set[start_id] = nstart
    import heapq

    priority_queue: List[Tuple[float, int]] = []
    heapq.heappush(priority_queue, (calc_cost(nstart, h_rs, h_dp, ngoal, config), start_id))
    u_list, d_list = calc_motion_inputs()
    while True:
        if not open_set:
            print("Error: Cannot find path, No open set")
            return None, None, None
        while priority_queue:
            _, current_id = heapq.heappop(priority_queue)
            if current_id in open_set:
                break
        else:
            print("Error: Priority queue exhausted without open nodes")
            return None, None, None
        current = open_set[current_id]
        updated, current = update_node_with_analytic_expansion(
            current, ngoal, obmap, config, kdtree, ox, oy
        )
        if updated:
            closed_set[calc_index(ngoal, config)] = current
            break
        open_set.pop(current_id, None)
        closed_set[current_id] = current
        for steer_input, direction_multiplier in zip(u_list, d_list):
            node = calc_next_node(
                current,
                current_id,
                steer_input,
                direction_multiplier,
                config,
                gkdtree,
            )
            if not verify_index(node, obmap, config, kdtree, ox, oy):
                continue
            node_id = calc_index(node, config)
            if node_id in closed_set:
                continue
            if node_id not in open_set:
                open_set[node_id] = node
                heapq.heappush(
                    priority_queue,
                    (calc_cost(node, h_rs, h_dp, ngoal, config), node_id),
                )
    rx, ry, ryaw = get_final_path(closed_set, ngoal, nstart, config)
    return rx, ry, ryaw


def main():  # pragma: no cover - plotting harness
    import matplotlib.pyplot as plt

    sx, sy, syaw = 20.0, 20.0, math.radians(90.0)
    gx, gy, gyaw = 180.0, 100.0, math.radians(-90.0)
    ox: List[float] = []
    oy: List[float] = []
    for i in range(201):
        ox.append(float(i))
        oy.append(0.0)
    for i in range(121):
        ox.append(200.0)
        oy.append(float(i))
    for i in range(201):
        ox.append(float(i))
        oy.append(120.0)
    for i in range(121):
        ox.append(0.0)
        oy.append(float(i))
    for i in range(81):
        ox.append(40.0)
        oy.append(float(i))
    for i in range(81):
        ox.append(80.0)
        oy.append(120.0 - float(i))
    for i in range(41):
        ox.append(120.0)
        oy.append(120.0 - float(i))
        ox.append(120.0)
        oy.append(float(i))
    for i in range(81):
        ox.append(160.0)
        oy.append(120.0 - float(i))
    start_time = time.perf_counter()
    rx, ry, ryaw = calc_hybrid_astar_path(
        sx,
        sy,
        syaw,
        gx,
        gy,
        gyaw,
        ox,
        oy,
        XY_GRID_RESOLUTION,
        YAW_GRID_RESOLUTION,
        OB_MAP_RESOLUTION,
    )
    elapsed = time.perf_counter() - start_time
    print(f"Hybrid A* planning completed in {elapsed:.2f} s")
    plt.plot(ox, oy, ".k", label="obstacles")
    if rx is not None:
        plt.plot(rx, ry, "-r", label="Hybrid A* path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
