# Mobile Robots: Navigation and Pathfinding Algorithms

This repository contains Python implementations for the navigation and control of a Robotino (Festo) holonomic platform. The project tracks the development of navigation systems from basic reactive methods to complex path-planning algorithms in dynamic environments.

## 1. Reactive Obstacle Avoidance (Lab 1)
Objective: Implement a reactive navigation system using the Potential Fields Method.

Methodology: The robot is driven by a virtual "Attractive Force" toward the target and a "Repulsive Force" away from obstacles.

Computer Vision: Developed a system using OpenCV and ArUco markers (Markers 1, 2, 3, 4) to calibrate a 210x210 cm field.

Key Logic: Perspective transformation was applied to obtain a top-down orthogonal view for accurate coordinate calculation.

## 2. Path Planning in Static Environments (Lab 2)
Objective: Compare fundamental pathfinding algorithms on a discretized grid.

Algorithms Implemented:

A* (A-Star): Optimized search using the Manhattan distance heuristic.

Dijkstra: Finds the shortest path based on edge weights.

BFS (Breadth-First Search): Finds the shortest path in terms of the number of nodes.

Grid Mapping: The environment is converted into a grid where cells are marked as "occupied" (1) or "free" (0) based on the robot's safety margin.

## 3. Dynamic Path Planning (Lab 3)
Objective: Adapt pathfinding for environments with moving obstacles.

Real-time Recalculation: Unlike static planning, the algorithm recalculates the entire path after every frame to account for moving ArUco markers.

Safety Features: Implemented a "Wait-and-Solve" logic. If a path is completely blocked by moving obstacles, the robot stops and waits for a valid path to open up.


# Мобильные роботы: Алгоритмы навигации и поиска пути
В данном репозитории представлены реализации на языке Python для навигации и управления голономной платформой Robotino (Festo). Проект охватывает развитие систем навигации: от базовых реактивных методов до сложных алгоритмов планирования пути в динамических средах.

## 1. Реактивный метод объезда препятствий (Лаб. №1)
Цель: Реализовать систему реактивной навигации с использованием метода потенциальных полей.

Методология: Робот движется под воздействием виртуальной «силы притяжения» к цели и «силы отталкивания» от препятствий.

Компьютерное зрение: Разработана система с использованием OpenCV и ArUco-маркеров (ID 1, 2, 3, 4) для калибровки игрового поля размером 210x210 см.

Ключевая логика: Применено перспективное преобразование для получения ортогонального вида сверху, что обеспечило точное вычисление координат.

## 2. Планирование пути в статических средах (Лаб. №2)
Цель: Сравнительный анализ фундаментальных алгоритмов поиска пути на дискретизированной сетке.

Реализованные алгоритмы:

A (A-Star):* Оптимизированный поиск с использованием эвристики Манхэттенского расстояния.

Алгоритм Дейкстры: Поиск кратчайшего пути с учетом весов ребер.

BFS (Поиск в ширину): Поиск кратчайшего пути по количеству узлов.

Картографирование: Среда преобразуется в сетку (grid), где ячейки помечаются как «занятые» (1) или «свободные» (0) с учетом радиуса безопасности робота.

## 3. Динамическое планирование пути (Лаб. №3)
Цель: Адаптация алгоритмов поиска пути для сред с движущимися препятствиями.

Пересчет в реальном времени: В отличие от статического планирования, алгоритм перестраивает весь маршрут после каждого кадра, учитывая перемещение ArUco-маркеров.

Функции безопасности: Реализована логика «Wait-and-Solve» (Ждать и решать). Если путь полностью заблокирован движущимися объектами, робот останавливается и ждет появления нового возможного маршрута.
