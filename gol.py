import colour
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
import tqdm
import video_writer
import cupy as cp
from cupyx.scipy import signal

# CONFIG
# VIDEO_WIDTH = 3840
VIDEO_WIDTH = 1000
# VIDEO_HEIGHT = 2160
VIDEO_HEIGHT = 2160
SECS = int(10)  # 3 mins 30 secs.
PIXEL_SIZE = 3
OUTPUT_PATH = 'videos/youtube-3m-30s-6px.mp4'
FPS = 60  # Frames per second.
HIGH_QUALITY = True
STATE_WIDTH = VIDEO_WIDTH // PIXEL_SIZE
STATE_HEIGHT = VIDEO_HEIGHT // PIXEL_SIZE
MAX_STEPS = 100000

# `RULE` specifies which cellular automaton rule to use.
RULE = 30

# `X_OFFSET` specifies how far from the center to place the initial first pixel.
X_OFFSET = 0

# These settings can be used for rule 110, which only grows to the left, so we
# offset the starting pixel to be close to the right edge of the screen.
# RULE = 110
# X_OFFSET = VIDEO_WIDTH // PIXEL_SIZE // 2 - 1 - 60 * 4

# The Game of Life state wraps across the left and right edges of the state,
# and dies out at the top of the state (all values of the top row are zero).
# By adding padding to the state, you extend the state beyond the edges of the
# visible window, essentially hiding the wrapping and/or dying out aspects of
# the state.
GOL_STATE_WIDTH_PADDING = VIDEO_WIDTH
GOL_STATE_HEIGHT_PADDING = VIDEO_HEIGHT


class Rule30AndGameOfLife:
    def __init__(self, width, height,
                 gol_percentage=0.5,
                 num_frames=MAX_STEPS):
        self.width = width
        self.height = height

        self.gol_height = int(height * gol_percentage)
        self.gol_state_width = self.width + GOL_STATE_WIDTH_PADDING * 2
        self.gol_state_height = self.gol_height + GOL_STATE_HEIGHT_PADDING

        self.gol_state = cp.zeros((self.gol_state_height, self.gol_state_width),
                                  cp.uint8)

        self.row_padding = num_frames // 2
        self.row_width = self.gol_state_width + self.row_padding * 2
        self.row = cp.zeros(self.row_width, cp.uint8)
        self.row[self.row_width // 2 + X_OFFSET] = 1

        self.rows_height = self.height - self.gol_height
        self.rows = cp.concatenate((
            cp.zeros((self.rows_height - 1, self.gol_state_width), cp.uint8),
            self.row[None, self.row_padding:-self.row_padding]
        ))

        self.row_neighbors = cp.array([1, 2, 4], dtype=cp.uint8)
        self.gol_neighbors = cp.array([[1, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 1]], dtype=cp.uint8)
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
            color_list += list(colour.Color(hex_colors[i]).range_to(
                colour.Color(hex_colors[i + 1]), color_decay_times[i]))
        color_list += [colour.Color('black')]
        rgb_list = [c.rgb for c in color_list]

        self.colors = (cp.array(rgb_list, float) * 255).astype(cp.uint8)

        self.decay = cp.full((self.height, self.width), len(self.colors) - 1,
                             int)

        self.rgb = None

        self.update_decay()
        self.update_rgb()

    def step(self):
        self.update_state()
        self.update_decay()
        self.update_rgb()

    def update_rule_kernel(self):
        self.rule_kernel = cp.array([int(x) for x in f'{self.rule:08b}'[::-1]], cp.uint8)

    def update_state_gpu(self):
        # Update `rows` (the state of the 2D cellular automaton).
        rule_index = signal.convolve2d(cp.asarray(self.row[None, :]),
                                       cp.asarray(self.row_neighbors[None, :]),
                                       mode='same', boundary='wrap')
        self.row = self.rule_kernel[rule_index[0].get()]
        transfer_row = self.rows[:1]
        self.rows = cp.concatenate((
            self.rows[1:],
            self.row[None, self.row_padding:-self.row_padding]
        ))

        # Update `gol_state` (the state of the 3D cellular automaton).
        num_neighbors = signal.convolve2d(self.gol_state, self.gol_neighbors,
                                          mode='same', boundary='wrap')
        self.gol_state = cp.logical_or(num_neighbors == 3,
                                       cp.logical_and(num_neighbors == 2,
                                                      self.gol_state)
                                       ).astype(cp.uint8)

        self.gol_state = cp.concatenate((
            cp.zeros((1, self.gol_state_width), cp.uint8),
            self.gol_state[1:-1],
            transfer_row
        ))

    def update_state(self):
        # Update `rows` (the state of the 2D cellular automaton).
        rule_index = signal.convolve2d(self.row[None, :],
                                       self.row_neighbors[None, :],
                                       mode='same', boundary='wrap')
        self.row = self.rule_kernel[rule_index[0]]
        transfer_row = self.rows[:1]
        self.rows = cp.concatenate((
            self.rows[1:],
            self.row[None, self.row_padding:-self.row_padding]
        ))

        # Update `gol_state` (the state of the 3D cellular automaton).
        num_neighbors = signal.convolve2d(self.gol_state, self.gol_neighbors,
                                          mode='same', boundary='wrap')
        self.gol_state = cp.logical_or(num_neighbors == 3,
                                       cp.logical_and(num_neighbors == 2,
                                                      self.gol_state)
                                       ).astype(cp.uint8)

        self.gol_state = cp.concatenate((
            cp.zeros((1, self.gol_state_width), cp.uint8),
            self.gol_state[1:-1],
            transfer_row
        ))

    def update_decay(self):
        visible_state = cp.concatenate(
            (self.gol_state[-self.gol_height:,
             GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING],
             self.rows[:, GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING]),
            axis=0)
        self.decay += 1
        self.decay = cp.clip(self.decay, None, len(self.colors) - 1)
        self.decay *= 1 - visible_state

    def update_rgb(self):
        self.rgb = self.colors[self.decay]


def main():
    # writer = video_writer.Writer(fps=FPS, high_quality=HIGH_QUALITY)
    animation = Rule30AndGameOfLife(STATE_WIDTH, STATE_HEIGHT)

    for _ in tqdm.trange(MAX_STEPS):
        small_frame = animation.rgb
        #enlarged_frame = imutils.resize(small_frame, VIDEO_WIDTH, VIDEO_HEIGHT, cv2.INTER_NEAREST)
        cv2.imshow("CA", small_frame)
        cv2.waitKey(1)

        # writer.add_frame(enlarged_frame)
        animation.step()
    # writer.write(OUTPUT_PATH)


if __name__ == '__main__':
    main()
