"""Synthesizes images with LIA."""

import os
import sys
import argparse
import cv2
from tqdm import tqdm
import tensorflow as tf
import numpy as np


from training.misc import load_pkl
from utils import imwrite, immerge, linear_interpolate
import dnnlib
import dnnlib.tflib as tflib


def parse_args():
    """Parses arguments."""
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument('--restore_path', type=str, default='',
                        help='The pre-trained encoder pkl file path')
    parser.add_argument("--batch_size", type=int,
                        default=8, help="size of the input batch")
    parser.add_argument("--data_dir_encode", type=str, default='',
                        help="Location of the encoded data")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             '`./outputs/rebuild_encodings` will be used by default.')
    parser.add_argument('-b', '--boundary_path', type=str, required=True,
                        help='Path to the semantic boundary. (required)')


    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')
    
    parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
    parser.add_argument('--steps', type=int, default=10,
                      help='Number of steps for image editing. (default: 10)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf_config = {'rnd.np_random_seed': 1000}
    tflib.init_tf(tf_config)
    assert os.path.exists(args.restore_path)
    _, _, _, Gs, _ = load_pkl(args.restore_path)
    num_layers = Gs.components.synthesis.input_shape[1]

    
    w_codes = np.load(args.data_dir_encode)
    boundary = np.load(args.boundary_path)
    print(w_codes.shape)
    latent_dim = w_codes.shape[1]
    total_num = w_codes.shape[0]
    
    # Building graph
    w_vec = tf.placeholder('float32', [None, latent_dim], name='w_codes')
    print(f'W in tensorflow graph: {w_vec.shape}')
    encoder_w_tile = tf.tile(w_vec[:, np.newaxis], [1, num_layers, 1])
    print(f'encoder_w_tile size: {encoder_w_tile.shape}')
    reconstructor = Gs.components.synthesis.get_output_for(encoder_w_tile, randomize_noise=False)
    sess = tf.get_default_session()

    save_dir = args.output_dir or './outputs/edited_images'
    os.makedirs(save_dir, exist_ok=True)
    
    for sample_id in tqdm(range(total_num), leave=True):
        #get edited codes for one image
        interpolations = linear_interpolate(w_codes[sample_id:sample_id+1],
                                            boundary,
                                            start_distance=args.start_distance,
                                            end_distance=args.end_distance,
                                            steps=args.steps)
        print(f'Interpolations= {interpolations.shape}')
        if interpolations[0].all == interpolations[1].all:
            print('STOP')
            sys.exit(1)
        interpolation_id = 0
        samples = sess.run(reconstructor, {w_vec: interpolations})
        samples = samples.transpose(0, 2, 3, 1)
        print(f'Samples= {samples.shape}')
        for image in samples:
            save_path = os.path.join(args.output_dir, f'{sample_id:03d}_{interpolation_id:03d}.jpg')
            imwrite(image, save_path)
            interpolation_id += 1
        print(f'interpolation_id: {interpolation_id}, steps: {args.steps}')
        assert interpolation_id == args.steps



if __name__ == "__main__":
    main()