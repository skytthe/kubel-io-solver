import os
from datetime import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.animation import PillowWriter
from textwrap import dedent
import cProfile
# from scipy.ndimage import label
from numba import njit


color_to_hex = {
    '🟥': '#d93030',  # Red
    '🟩': '#33FF57',  # Green
    '🟦': '#3385FF',  # Blue
    '🟨': '#ffdd24',  # Yellow
    '🟪': '#A020F0',  # Purple
    '🟫': '#8B4513',  # Brown
    '⬜': '#808080',  # Gray
    '🟧': '#ff9100'   # Orange
}


def move(grid, index, axis, steps):
    new_grid = np.copy(grid)
    if axis == 0:
        new_grid[index] = np.roll(new_grid[index], steps)
    else:
        new_grid[:, index] = np.roll(new_grid[:, index], steps)
    return new_grid


def serialize(grid):
    # return tuple(map(tuple, grid.tolist()))
    return grid.tobytes()

@njit
def get_cost(grid, tmp):
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




def solve_beam(grid, ncolors, beam_width=1000, max_depth=200):
    start_time = time.time()

    n = grid.shape[0]

    initial_state = {
        'grid': grid,
        'moves': [],
        'cost': get_cost(grid, ncolors)
    }
    beam = [initial_state]

    visited = {serialize(grid)}

    depth = 0
    while depth < max_depth:
        new_beam = []
        for state in beam:
            for i in range(n):
                for axis in [0, 1]:
                    for steps in range(1, n): 
                        new_grid = move(state['grid'], i, axis, steps)
                        ser = serialize(new_grid)
                        new_moves = state['moves'] + [(i, axis, steps)]
                        new_cost = get_cost(new_grid, ncolors)
                        if ser not in visited:
                            visited.add(ser)
                            new_beam.append({
                                'grid': new_grid,
                                'moves': new_moves,
                                'cost': new_cost
                            })
        if not new_beam:
            break

        new_beam.sort(key=lambda s: (s['cost'], len(s['moves'])))

        beam = new_beam[:beam_width]
        depth += 1
        best_cost = beam[0]['cost']
        print(f"Depth {depth}: best cost = {best_cost}, beam size = {len(beam)}")
        if best_cost == 0:
            solution_moves = beam[0]['moves']
            print(f"Solved at depth {depth} in {time.time()-start_time:.2f} seconds")
            print(f"Solution moves: {solution_moves}")
            print(f"Number of moves: {len(solution_moves)}")
            return solution_moves

    print("No solution found within max depth")
    return None


def apply_moves(grid, moves):
    states = [grid.copy()]
    current = grid.copy()
    for move_tuple in moves:
        i, axis, steps = move_tuple
        current = move(current, i, axis, steps)
        states.append(current.copy())
    return states

def animate_solution(grid, solution_moves, colors):
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    folder_path = "solutions"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = f"solutions/{formatted_time}"
    os.makedirs(folder_path)

    states = apply_moves(grid, solution_moves)

    cmap = ListedColormap([color_to_hex[colors[i]]
                        for i in range(len(colors) // 2)])

    for frame, state in enumerate(states):
        fig, ax = plt.subplots()
        im = ax.imshow(state, cmap=cmap)
        ax.set_title(f"Kubel Puzzle - Move {frame}")
        if frame == 0:
            move_text = "Start"
        else:
            move_text = f"Move {frame}: {solution_moves[frame-1]}"
        ax.text(0.05, 0.95, move_text, transform=ax.transAxes,
                color='w', fontsize=12, verticalalignment='top')
        plt.savefig(os.path.join(folder_path, f"kubel_move_{frame:03d}.png"))
        plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(states[0], cmap=cmap)
    ax.set_title("Kubel Puzzle - Move 0 (Start)")
    move_text_obj = ax.text(0.05, 0.95, "", transform=ax.transAxes,
                            color='w', fontsize=12, verticalalignment='top')

    def update(frame):
        im.set_data(states[frame])
        if frame == 0:
            move_text_obj.set_text("Start")
        else:
            move_text_obj.set_text(
                f"Move {frame}: {solution_moves[frame-1]}")
        ax.set_title(f"Kubel Puzzle - Move {frame}")
        return im, move_text_obj

    ani = animation.FuncAnimation(fig, update, frames=len(
        states), interval=1000, blit=True, repeat_delay=4000)

    ani.save(os.path.join(folder_path, "kubel_solution.gif"), writer=PillowWriter(fps=1))

    plt.show()

if __name__ == '__main__':
    # colors: 🟥 🟩 🟦 🟨 🟪 🟫 ⬜ 🟧
    input = dedent("""\
        🟥🟩🟨🟧🟥🟪
        🟩🟪🟨🟥🟫🟩
        🟪🟥🟨🟧🟩🟨
        🟨🟥🟧🟩🟧🟩
        🟫🟫🟩🟧🟧🟫
        🟧🟧🟧🟪🟪🟧
    """)

    input = dedent("""\
        ⬜🟨🟪🟥
        🟫🟩🟦🟨
        🟦🟪⬜🟦
        🟥🟧🟨🟪
    """)

    input = dedent("""\
        🟦🟧🟧🟧🟦🟦🟧
        🟨🟧🟨🟦🟨🟧🟦
        🟦🟦🟨🟨🟧🟨🟧
        🟦🟧🟧🟦🟦🟦🟧
        🟨🟦🟧🟦🟨🟧🟨
        🟨🟨🟦🟨🟧🟦🟧
        🟨🟧🟨🟧🟨🟧🟦
    """)    

    colors = {i: char for i, char in enumerate(
        {char for char in input if char != '\n'})}
    ncolors = len(colors.items())

    colors.update({char: i for i, char in colors.items()})

    grid = np.array([[colors[char] for char in line]
                    for line in input.splitlines()])

    # cProfile.run('solve_beam(grid,ncolor, beam_width=1000, max_depth=20)')

    solution_moves = solve_beam(grid, ncolors, beam_width=1400, max_depth=20)

    if solution_moves is not None:
        animate_solution(grid, solution_moves, colors)

        
