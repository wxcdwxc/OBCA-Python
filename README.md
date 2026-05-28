# OBCA-Python

本项目基于 [OBCA](https://github.com/XiaojingGeorgeZhang/OBCA)（Optimization-Based Collision Avoidance）原项目，将泊车规划部分的 [Julia](https://julialang.org/) 实现改写为 **Python**，并在原有单车+单挂车的基础上，**增加了双挂车（双挂）的轨迹规划实现**。

原项目论文请见 [arXiv:1711.03449](http://arxiv.org/abs/1711.03449)。

## 与原项目的差异

- **语言迁移**：将 Julia 泊车规划代码移植为 Python，便于更多开发者使用和扩展
- **双挂车支持**：在原有 Truck-Trailer 的基础上，新增了双挂车（Truck with Two Trailers）的碰撞约束建模与轨迹优化
- **可视化**：使用 Pygame / Matplotlib 实现轨迹可视化

## 文件说明

| 文件 | 说明 |
|------|------|
| `TrailerParkingOptimization.py` | 单挂车泊车轨迹优化 |
| `TrailerParkingOptimization2.py` | 单挂车泊车轨迹优化2 |
| `TrailerParkingOptimization3.py` | 双挂车泊车轨迹优化 |
| `ManualDrive.py` | 手动控制界面 |
| `DualMultWS.py` | 双挂车碰撞约束建模 |
| `obstHrep.py` | 障碍物 H-表示（half-plane representation） |
| `A_star/hybrid_a_star.py` | Hybrid A* 路径搜索 |
| `PyGamePlot*.py` | Pygame 可视化绘图 |

## 运行环境

依赖：`numpy`, `scipy`, `casadi`, `pygame`, `matplotlib`

## 致谢

本项目源于 [XiaojingGeorgeZhang/OBCA](https://github.com/XiaojingGeorgeZhang/OBCA) 的开源工作，感谢原作者的贡献。

