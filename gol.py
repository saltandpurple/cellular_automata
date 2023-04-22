import colour
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
import tqdm
import cupy as cp
from cupyx.scipy import signal

# TODO: cleanup superfluous params
# VIDEO CONFIG
# VIDEO_WIDTH = 3840
# VIDEO_HEIGHT = 2160
# SECS = int(10)  # 3 mins 30 secs.
# PIXEL_SIZE = 1
# OUTPUT_PATH = 'videos/youtube-3m-30s-6px.mp4'
# FPS = 60  # Frames per second.
# HIGH_QUALITY = True

# These settings can be used for rule 110, which only grows to the left, so we
# offset the starting pixel to be close to the right edge of the screen.
# RULE = 110
# X_OFFSET = VIDEO_WIDTH // PIXEL_SIZE // 2 - 1 - 60 * 4

# The Game of Life state wraps across the left and right edges of the state,
# and dies out at the top of the state (all values of the top row are zero).
# By adding padding to the state, you extend the state beyond the edges of the
# visible window, essentially hiding the wrapping and/or dying out aspects of
# the state.
STATE_WIDTH = 4000
STATE_HEIGHT = 1500
GOL_STATE_WIDTH_PADDING = STATE_WIDTH
GOL_STATE_HEIGHT_PADDING = STATE_HEIGHT

GOL_PERCENTAGE = 0.9  # The part of the screen that is made up by the GOL (the rest is the rule feed preview)
MAX_STEPS = 1200000  # How long to run
DISPLAY_INTERVAL = 1000  # How often to visually update the state
ENTROPY_ENABLED = False
ENTROPY_FACTOR = 0.999999  # The randomness in the grid state - a higher number means lower entropy! 0 means total randomness, 1 no randomness

RULE = 30  # `RULE` specifies which cellular automaton rule to use.
X_OFFSET = 0  # `X_OFFSET` specifies how far from the center to place the initial first pixel.


class Rule30AndGameOfLife:
    def __init__(self, width, height,
                 gol_percentage=GOL_PERCENTAGE,
                 num_frames=MAX_STEPS):
        self.width = width
        self.height = height

        self.gol_height = int(height * gol_percentage)
        self.gol_state_width = self.width + GOL_STATE_WIDTH_PADDING * 2  # Create gol width and add padding
        self.gol_state_height = self.gol_height  # No padding added in gol height, so we can see the whole state

        self.gol_state = cp.zeros((self.gol_state_height, self.gol_state_width), np.uint8)
        self.entropy = cp.zeros((self.gol_state_height, self.gol_state_width))

        self.row_padding = num_frames // 2
        self.row_width = self.gol_state_width + self.row_padding * 2
        self.row = cp.zeros(self.row_width, np.uint8)
        self.row[self.row_width // 2 + X_OFFSET] = 1

        self.rule_feed_height = self.height - self.gol_height
        self.rule_feed = cp.concatenate((
            cp.zeros((self.rule_feed_height - 1, self.gol_state_width), np.uint8),
            self.row[None, self.row_padding:-self.row_padding]
        ))

        self.row_neighbors = cp.array([1, 2, 4], dtype=np.uint8)
        self.gol_neighbors = cp.array([[1, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 1]], dtype=np.uint8)
        self.rule = RULE
        self.rule_kernel = None
        self.update_rule_kernel()

        hex_colors = [
            '#711c91',
            '#ea00d9',
            '#0abdc6',
            '#133e7c',
            '#091833',
            '#000103'
        ]
        color_decay_times = [2 * 8 ** i for i in range(len(hex_colors) - 1)]
        assert len(hex_colors) == len(color_decay_times) + 1
        color_list = [colour.Color('white')]
        for i in range(len(hex_colors) - 1):
            color_list += list(colour.Color(hex_colors[i]).
                               range_to(colour.Color(hex_colors[i + 1]), color_decay_times[i]))
        color_list += [colour.Color('black')]
        rgb_list = [c.rgb for c in color_list]

        # this must be done with numpy, if display output is to be shown
        self.colors = (cp.array(rgb_list, float) * 255).astype(np.uint8)
        self.decay = cp.full((self.height, self.width), len(self.colors) - 1, int)
        self.rgb = None

        self.update_decay()
        self.update_rgb()

    def step(self):
        self.update_rule_feed()
        self.update_gol_state()
        self.update_decay()
        self.update_rgb()

    def update_rule_kernel(self):
        self.rule_kernel = cp.array([int(x) for x in f'{self.rule:08b}'[::-1]], np.uint8)

    # Generate another rule feed row and add it to its bottom. Remove the first one.
    def update_rule_feed(self):
        rule_index = signal.convolve2d(self.row[None, :],
                                       self.row_neighbors[None, :],
                                       mode='same', boundary='wrap')
        self.row = self.rule_kernel[rule_index[0]]
        self.rule_feed = cp.concatenate((
            self.rule_feed[1:],
            self.row[None, self.row_padding:-self.row_padding]
        ))

    """ 
    Determine the new state of the Cellular automaton.
     
    First, apply the CA ruleset. We currently use Conway's Game Of Life ruleset: 
    A cell survives, if it has 2 or 3 neighbors. A cell is born, if it has exactly 3 neighbors. Otherwise, the cell dies.
    Then, generate the entropy grid which is consequently used to partially randomize the gol state.
    This currently works as follows:
    With a chance of ENTROPY_FACTOR, flip a cell of the rulegrid to its opposite state.
    This is intended to break up stale constellations.
    """
    def update_gol_state(self):
        # Update GOL state
        num_neighbors = signal.convolve2d(self.gol_state, self.gol_neighbors, mode='same', boundary='wrap')
        self.gol_state = cp.logical_or(num_neighbors == 3, cp.logical_and(num_neighbors == 2, self.gol_state)).astype(cp.uint8)

        # Generate and apply entropy
        if ENTROPY_ENABLED:
          self.entropy = cp.random.rand(self.gol_state_height, self.gol_state_width)
          self.gol_state = cp.logical_xor(cp.logical_and(self.entropy < ENTROPY_FACTOR, self.gol_state == 1), cp.logical_and(self.entropy >= ENTROPY_FACTOR, self.gol_state == 0)).astype(cp.uint8)

        # Concatenate empty row, GOL state (minus top and bottom row) and the feed row
        feed_row = self.rule_feed[:1]
        self.gol_state = cp.concatenate((
            cp.zeros((1, self.gol_state_width), cp.uint8),
            self.gol_state[1:-1],
            feed_row
        ))

    # Glue the feed and gol state together and apply the (purely visual) decay function
    def update_decay(self):
        visible_state = cp.concatenate(
            (self.gol_state[:, GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING],
             self.rule_feed[:, GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING]),
            axis=0)

        self.decay += 1
        self.decay = cp.clip(self.decay, None, len(self.colors) - 1)
        self.decay *= 1 - visible_state

    def update_rgb(self):
        self.rgb = self.colors[self.decay]

    # Display the current state of the given animation for duration milliseconds
    @staticmethod
    def display_state(animation, duration):
        state_in_rgb = cv2.cvtColor(cp.asnumpy(animation.rgb), cv2.COLOR_BGR2RGB)
        # We need to convert BGR to OpenCVs RGB
        cv2.imshow("CA", state_in_rgb)
        cv2.waitKey(duration)


def main():
    # writer = video_writer.Writer(fps=FPS, high_quality=HIGH_QUALITY)
    animation = Rule30AndGameOfLife(STATE_WIDTH, STATE_HEIGHT)
    cv2.namedWindow("CA", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO )

    # Step through the animation and display the current state every DISPLAY_INTERVAL steps
    for step in tqdm.trange(MAX_STEPS):
        if step % DISPLAY_INTERVAL == 0:
            Rule30AndGameOfLife.display_state(animation, 1)
        animation.step()

    # Display the final state until key pressed
    Rule30AndGameOfLife.display_state(animation, 0)


if __name__ == '__main__':
    main()
