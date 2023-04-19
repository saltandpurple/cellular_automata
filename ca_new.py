import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

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

def step(grid):
    new_grid = np.zeros_like(grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            neighbors = count_neighbors(grid, x, y)
            new_grid[y, x] = game_of_life_rules(neighbors, grid[y, x])
    return new_grid

def update(frame, img, grid):
    new_grid = step(grid)
    img.set_array(new_grid)
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
grid_size = (100, 100)
grid = initialize_grid(grid_size, init_type='random')

# Create a colormap with purple, red, and yellow
cmap = ListedColormap(['#329BA5', '#711c91', '#ea00d9', '#ff0000', '#ff6600', '#ffcc00', '#ffff00'])

# Set up the plot
fig, ax = plt.subplots()
img = ax.imshow(color_neighbors(grid), cmap=cmap, interpolation="nearest")
plt.axis("off")

# Animate and display the simulation
def update_colored(frame, img, grid):
    new_grid = step(grid)
    img.set_array(color_neighbors(new_grid))
    grid[:] = new_grid[:]
    return img,

ani = animation.FuncAnimation(fig, update_colored, fargs=(img, grid), frames=200, interval=100, blit=True)
plt.show()