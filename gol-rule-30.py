import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import time

def game_of_life_rules(neighbors, state):
    if state == 1:
        return 1 if neighbors in [2, 3] else 0
    else:
        return 1 if neighbors == 3 else 0

def count_neighbors(grid, x, y):
    neighbors = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                neighbors += grid[(y + dy) % grid.shape[0], (x + dx) % grid.shape[1]]
    return neighbors

def step(grid, rule_30_side='left'):
    new_grid = np.zeros_like(grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            neighbors = count_neighbors(grid, x, y)
            new_grid[y, x] = game_of_life_rules(neighbors, grid[y, x])

    # Rule 30
    new_row = np.zeros(grid.shape[1], dtype=int)
    for x in range(1, grid.shape[1] - 1):
        pattern = grid[0, x - 1] * 4 + grid[0, x] * 2 + grid[0, x + 1]
        new_row[x] = 1 if (30 >> pattern) & 1 else 0

    if rule_30_side == 'left':
        new_grid[0, :] = new_row
    elif rule_30_side == 'right':
        new_grid[-1, :] = new_row[::-1]

    return new_grid

def update(frame, img, grid, rule_30_side):
    new_grid = step(grid, rule_30_side)
    img.set_array(color_neighbors(new_grid))
    grid[:] = new_grid[:]
    return img,

def initialize_grid(grid_size, init_type='empty'):
    if init_type == 'empty':
        grid = np.zeros(grid_size, dtype=int)
    elif init_type == 'random':
        grid = np.random.choice([0, 1], size=grid_size)
    return grid

def color_neighbors(grid):
    colored_grid = np.zeros_like(grid, dtype=int)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            neighbors = count_neighbors(grid, x, y)
            colored_grid[y, x] = neighbors + 1 if grid[y, x] else 0
    return colored_grid


# Set up the grid and initial state
GRID_SIZE = (200, 200)
FIG_SIZE = (10, 10)

grid = initialize_grid(GRID_SIZE, init_type='random')
rule_30_side = 'left'

# Create a colormap with purple, red, and yellow
cmap = ListedColormap(['#329BA5', '#711c91', '#ea00d9', '#ff0000', '#ff6600', '#ffcc00', '#ffff00'])

# Set up the plot
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
img = ax.imshow(color_neighbors(grid), cmap=cmap, interpolation="nearest", aspect="auto")
plt.axis("off")

# Animate and display the simulation
step_interval = 100  # milliseconds per step (1000 ms / 10 steps/s)
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, rule_30_side), interval=step_interval, blit=True)
plt.show()
