import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import imageio
import numpy as np
from tqdm import tqdm

import argparse, os, subprocess
from os.path import join

from osrt.utils.visualize import setup_axis, background_image

def compile_video_plot(path, frames=False, num_frames=1000000000):

    frame_output_dir = os.path.join(path, 'frames')
    if not os.path.exists(frame_output_dir):
        os.mkdir(frame_output_dir)

    input_image = imageio.imread(join(path, 'input_0.png'))
    bg_image = background_image((240, 320, 3), gridsize=12)

    dpi = 100

    for frame_id in tqdm(range(num_frames)):
        if not frames:
            break

        fig, ax = plt.subplots(1, 3, figsize=(900/dpi, 350/dpi), dpi=dpi)

        plt.subplots_adjust(wspace=0.05, hspace=0.08, left=0.01, right=0.99, top=0.995, bottom=0.035)
        for cell in ax:
            setup_axis(cell)

        ax[0].imshow(input_image)
        ax[0].set_xlabel('Input Image 1')
        try:
            render = imageio.imread(join(path, 'renders', f'{frame_id}.png'))
        except FileNotFoundError:
            break
        ax[1].imshow(bg_image)
        ax[1].imshow(render[..., :3])
        ax[1].set_xlabel('Rendered Scene')
       
        segmentations = imageio.imread(join(path, 'segmentations', f'{frame_id}.png'))
        ax[2].imshow(segmentations)
        ax[2].set_xlabel('Segmentations')

        fig.savefig(join(frame_output_dir, f'{frame_id}.png'))
        plt.close()

        frame_id += 1

    frame_placeholder = join(frame_output_dir, '%d.png')
    video_out_file = join(path, 'video.mp4')
    print('rendering video to ', video_out_file)
    subprocess.call(['ffmpeg', '-y', '-framerate', '60', '-i', frame_placeholder,
                     '-pix_fmt', 'yuv420p', '-b:v', '1M', '-threads', '1', video_out_file])

def compile_video_render(path):
    frame_placeholder = os.path.join(path, 'renders', '%d.png')
    video_out_file = os.path.join(path, 'video-renders.mp4')
    subprocess.call(['ffmpeg', '-y', '-framerate', '60', '-i', frame_placeholder,
                     '-pix_fmt', 'yuv420p', '-b:v', '1M', '-threads', '1', video_out_file])

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Render a video of a scene.'
    )
    parser.add_argument('path', type=str, help='Path to image files.')
    parser.add_argument('--plot', action='store_true', help='Plot available data, instead of just renders.')
    parser.add_argument('--noframes', action='store_true', help="Assume frames already exist and don't rerender them.")
    args = parser.parse_args()

    if args.plot:
        compile_video_plot(args.path, frames=not args.noframes)
    else:
        compile_video_render(args.path)

