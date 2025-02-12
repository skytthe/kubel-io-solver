import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.animation import PillowWriter
from textwrap import dedent
import cProfile
from scipy.ndimage import label
from numba import njit


def connected_components_cost_bfs(grid, ncolors):
    n = grid.shape[0]
    total_cost = 0
    unique_colors = np.unique(grid)

    for color in unique_colors:
        visited = np.zeros((n, n), dtype=bool)
        components = 0
        for i in range(n):
            for j in range(n):
                if grid[i, j] == color and not visited[i, j]:
                    components += 1
                    queue = deque()
                    queue.append((i, j))
                    visited[i, j] = True
                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < n and 0 <= ny < n:
                                if grid[nx, ny] == color and not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    queue.append((nx, ny))
        total_cost += (components - 1)
    return total_cost


def connected_components_cost_scipy(grid, ncolors):
    structure = np.array([[0, 1, 0],  
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    total_components = 0

    for value in range(ncolors):
        binary_mask = (grid == value)
        labeled_array, num_components = label(binary_mask, structure=structure)
        total_components += num_components
    return total_components-ncolors


def connected_components_cost_bit(grid, tmp):
    n = grid.shape[0]
    total_cells = n * n

    mask_R = 0
    mask_L = 0 
    for i in range(n):
        mask_R |= (1 << (i * n + (n - 1)))
        mask_L |= (1 << (i * n + 0))
    full_mask = (1 << total_cells) - 1

    cost = 0

    for col in range(ncolors):
        bitboard = 0
        for i in range(n):
            for j in range(n):
                if grid[i, j] == col:
                    bit_index = i * n + j
                    bitboard |= (1 << bit_index)
        if bitboard == 0:
            continue

        comp_count = 0
        remaining = bitboard
        while remaining:
            seed = remaining & -remaining
            comp = seed
            while True:
                neighbors = (
                    ((comp & ~mask_R) << 1) |
                    ((comp & ~mask_L) >> 1) |
                    (comp << n) |
                    (comp >> n)
                ) & full_mask
                new_comp = comp | (neighbors & bitboard)
                if new_comp == comp:
                    break
                comp = new_comp
            comp_count += 1
            remaining &= ~comp

        cost += (comp_count - 1)

    return cost


if __name__ == '__main__':
    # colors: 🟥 🟩 🟦 🟨 🟪 🟫 ⬜ 🟧
    input1 = dedent("""\
        🟥🟩🟨🟧🟥🟪
        🟩🟪🟨🟥🟫🟩
        🟪🟥🟨🟧🟩🟨
        🟨🟥🟧🟩🟧🟩
        🟫🟫🟩🟧🟧🟫
        🟧🟧🟧🟪🟪🟧
    """)

    input2 = dedent("""\
        ⬜🟨🟪🟥
        🟫🟩🟦🟨
        🟦🟪⬜🟦
        🟥🟧🟨🟪
    """)

    input3 = dedent("""\
        🟦🟧🟧🟧🟦🟦🟧
        🟨🟧🟨🟦🟨🟧🟦
        🟦🟦🟨🟨🟧🟨🟧
        🟦🟧🟧🟦🟦🟦🟧
        🟨🟦🟧🟦🟨🟧🟨
        🟨🟨🟦🟨🟧🟦🟧
        🟨🟧🟨🟧🟨🟧🟦
    """)    

    for input in [input1,input2,input3]:

        colors = {i: char for i, char in enumerate(
            {char for char in input if char != '\n'})}
        ncolors = len(colors.items())
        print(f"number of colors: {ncolors}")
        colors.update({char: i for i, char in colors.items()})

        grid = np.array([[colors[char] for char in line]
                        for line in input.splitlines()])

        print("Grid>")
        for row in grid:
            print("\t", row)
        print(f"connected_components_cost_bfs:   {connected_components_cost_bfs(grid,ncolors)}")
        print(f"connected_components_cost_bit:   {connected_components_cost_bit(grid,ncolors)}")
        print(f"connected_components_cost_scipy: {connected_components_cost_scipy(grid,ncolors)}")
        print("\n\n")
