import colour
import cv2
import imutils
import numpy as np
import cupy as cp
import matplotlib.animation as animation
import tqdm
from cupyx.scipy import signal
from ruleset_gol import RulesetGameOfLife
from ruleset_mnca import RulesetMultipleNeighbourhoods


# These settings can be used for rule 110, which only grows to the left, so we
# offset the starting pixel to be close to the right edge of the screen.
# RULE = 110
# X_OFFSET = VIDEO_WIDTH // PIXEL_SIZE // 2 - 1 - 60 * 4

# The Game of Life state wraps across the left and right edges of the state,
# and dies out at the top of the state (all values of the top row are zero).
# By adding padding to the state, you extend the state beyond the edges of the
# visible window, essentially hiding the wrapping and/or dying out aspects of
# the state.
# TODO: cleanup the padding, we don't need it anymore
STATE_WIDTH = 4000 // 2
STATE_HEIGHT = 1500 // 2
GOL_STATE_WIDTH_PADDING = STATE_WIDTH
GOL_STATE_HEIGHT_PADDING = STATE_HEIGHT

# TODO: adjust the params to the new implementation style (switch between rulesets, enable/disable rulefeed etc)
INIT_MODE = "empty"
GOL_PERCENTAGE = 0.9  # The part of the screen that is made up by the GOL (the rest is the rule feed preview)
MAX_STEPS = 1200000  # How long to run
DISPLAY_INTERVAL = 100  # How often to visually update the state

RULE = 30  # `RULE` specifies which cellular automaton rule to use.
X_OFFSET = 0  # `X_OFFSET` specifies how far from the center to place the initial first pixel.


class CellularAutomaton:
    def __init__(self, width, height,
                 ca_percentage=GOL_PERCENTAGE,
                 num_frames=MAX_STEPS):
        self.width = width
        self.height = height

        self.ca_height = int(self.height * ca_percentage)
        self.ca_state_width = self.width + GOL_STATE_WIDTH_PADDING * 2  # Create ca width and add padding
        self.ca_state_height = self.ca_height  # No padding added in ca height, so we can see the whole state

        if INIT_MODE == "random":
            self.state = cp.random.random_integers(0, 1, (self.ca_state_height, self.ca_state_width)).astype(cp.uint8)
        else:
            self.state = cp.zeros((self.ca_state_height, self.ca_state_width), cp.uint8)

        self.row_padding = num_frames // 2
        self.row_width = self.ca_state_width + self.row_padding * 2
        self.row = cp.zeros(self.row_width, np.uint8)
        self.row[self.row_width // 2 + X_OFFSET] = 1

        self.rule_feed_height = self.height - self.ca_height
        self.rule_feed = cp.concatenate((
            cp.zeros((self.rule_feed_height - 1, self.ca_state_width), np.uint8),
            self.row[None, self.row_padding:-self.row_padding]
        ))

        self.row_neighbors = cp.array([1, 2, 4], dtype=np.uint8)

        # Choose your ruleset here
        self.ruleset = RulesetMultipleNeighbourhoods()
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
        self.bgr = None

        self.update_decay()
        self.update_rgb()

    def step(self):
        self.update_rule_feed()
        self.update_state()
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
    """
    def update_state(self):
        # Update ca_state according to the defined ruleset
        self.state = self.ruleset.calculate_next_state(self.state)

        # Concatenate empty row, GOL state (minus top and bottom row) and the feed row
        feed_row = self.rule_feed[:1]
        self.state = cp.concatenate((
            cp.zeros((1, self.ca_state_width), cp.uint8),
            self.state[1:-1],
            feed_row
        ))

    # Glue the feed and ca state together and apply the (purely visual) decay function
    def update_decay(self):
        visible_state = cp.concatenate(
            (self.state[:, GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING],
             self.rule_feed[:, GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING]),
            axis=0)

        self.decay = visible_state

        # TODO: rework this to be a) configurable and b) working with MNCAs
        # self.decay += 1
        # self.decay = cp.clip(self.decay, None, len(self.colors) - 1)
        # self.decay *= 1 - visible_state

    def update_rgb(self):
        self.bgr = self.colors[self.decay]

    # Display the current state of the given animation for milliseconds duration
    @staticmethod
    def display_state(animation, duration):
        # We need to convert BGR to OpenCVs RGB # TODO: rework this/check this
        state_in_rgb = cv2.cvtColor(cp.asnumpy(animation.bgr), cv2.COLOR_BGR2RGB)
        # state_in_rgb = cp.asnumpy(animation.bgr)
        cv2.imshow("CA", state_in_rgb)
        cv2.waitKey(duration)


def main():
    # writer = video_writer.Writer(fps=FPS, high_quality=HIGH_QUALITY)
    animation = CellularAutomaton(STATE_WIDTH, STATE_HEIGHT)
    cv2.namedWindow("CA", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Step through the animation and display the current state every DISPLAY_INTERVAL steps
    for step in tqdm.trange(MAX_STEPS):
        if step % DISPLAY_INTERVAL == 0:
            CellularAutomaton.display_state(animation, 1)
        animation.step()

    # Display the final state until key pressed
    CellularAutomaton.display_state(animation, 0)


if __name__ == '__main__':
    main()
