import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
import time

# Configuration
grid_size = (300, 300)
fig_size = (10, 10)
rule_30_input_side = "left"
initial_rate = 20  # Steps per second
step_interval = 1000 // initial_rate

def initialize_grid(grid_size, init_type='empty'):
    if init_type == 'empty':
        grid = np.zeros(grid_size, dtype=int)
    elif init_type == 'random':
        grid = np.random.choice([0, 1], size=grid_size)
    return grid
# Create the initial grid
grid = initialize_grid(grid_size, init_type='empty')

# Define the Game of Life update rule
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

# Define colors
cmap = plt.get_cmap("inferno")

# Initialize Rule 30 buffer with 100 steps
rule_30_steps = 300
rule_30_buffer = np.zeros((rule_30_steps, grid_size[0] - 1), dtype=np.uint8)

def generate_rule_30(rule_30_buffer):
    """
    Generate the Rule 30 automaton in a separate buffer.

    Args:
    rule_30_buffer: A numpy array containing the Rule 30 buffer.
    """
    # Rule 30 in binary
    rule_30 = 30
    binary_rule = np.array([int(x) for x in f'{rule_30:08b}'[::-1]], np.uint8)

    # Initialize the first row of the Rule 30 buffer
    rule_30_buffer[0, rule_30_buffer.shape[1] // 2] = 1

    for i in range(1, rule_30_buffer.shape[0]):
        # Calculate the next state for Rule 30 using bitwise operations
        rule_30_buffer[i, 1:-1] = binary_rule[(rule_30_buffer[i-1, :-2] << 2) | (rule_30_buffer[i-1, 1:-1] << 1) | rule_30_buffer[i-1, 2:]]

        # Handle edge cases
        rule_30_buffer[i, 0] = binary_rule[(rule_30_buffer[i-1, -1] << 2) | (rule_30_buffer[i-1, 0] << 1) | rule_30_buffer[i-1, 1]]
        rule_30_buffer[i, -1] = binary_rule[(rule_30_buffer[i-1, -2] << 2) | (rule_30_buffer[i-1, -1] << 1) | rule_30_buffer[i-1, 0]]

    # Flip the Rule 30 buffer horizontally
    rule_30_buffer = np.fliplr(rule_30_buffer)

    return rule_30_buffer


rule_30_buffer = generate_rule_30(rule_30_buffer)


def step(grid, rule_30_buffer):
    # Game of Life logic
    new_grid = np.zeros_like(grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            count = np.sum(grid[y-1:y+2, x-1:x+2]) - grid[y, x]
            if grid[y, x]:
                new_grid[y, x] = count in (2, 3)
            else:
                new_grid[y, x] = count == 3

    grid = new_grid

    # Update the Rule 30 buffer
    rule_30_buffer[:-1] = rule_30_buffer[1:]

    # Update the Rule 30 buffer using the binary_rule
    binary_rule = np.array([int(x) for x in f'{30:08b}'[::-1]], np.uint8)
    rule_30_buffer[-1, 1:-1] = binary_rule[(rule_30_buffer[-2, :-2] << 2) | (rule_30_buffer[-2, 1:-1] << 1) | rule_30_buffer[-2, 2:]]

    # Handle edge cases
    rule_30_buffer[-1, 0] = binary_rule[(rule_30_buffer[-2, -1] << 2) | (rule_30_buffer[-2, 0] << 1) | rule_30_buffer[-2, 1]]
    rule_30_buffer[-1, -1] = binary_rule[(rule_30_buffer[-2, -2] << 2) | (rule_30_buffer[-2, -1] << 1) | rule_30_buffer[-2, 0]]

    # TODO: fix this, so that it feeds correctly into the bottom
    grid[:-1, 0] = rule_30_buffer[:-1, 0]
    # print("buffer:\n", rule_30_buffer)
    # print("grid:\n", grid)
    # time.sleep(1)

    # Debug output
    # print("Grid [:-1, 0]:\n", grid[:-1, 0])
    # print("buffer [:-1, 0]:\n", rule_30_buffer[:-1, 0])
    return grid, rule_30_buffer


def update(*args):
    global grid, rule_30_buffer
    grid, rule_30_buffer = step(grid, rule_30_buffer)
    img.set_data(grid)
    return [img]  # return img as a list



fig, ax = plt.subplots(figsize=fig_size)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
img = ax.imshow(rule_30_buffer, cmap=cmap, interpolation="nearest", aspect="auto")
ani = animation.FuncAnimation(fig, update, interval=step_interval, blit=True, cache_frame_data=False)

# Add these lines to handle the AttributeError properly
try:
    plt.show()
except AttributeError as e:
  print("nothing")
    # if "_resize_id" not in str(e):
      # print("Warning: An AttributeError related to '_resize_id' occurred, but the program continues to run.")
