"""Synthesizes images with LIA."""

import os
import sys
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils import imwrite, immerge
from training.misc import load_pkl
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
    parser.add_argument("--data_dir_encod/e", type=str, default='',
                        help="Location of the encoded data")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             '`./outputs/rebuild_encodings` will be used by default.')
    parser.add_argument('--total_nums', type=int, default=5,
                        help='number of loops for sampling')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')

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

    
    batch_codes = np.load(args.data_dir_encode)
    print(batch_codes.shape)
    latent_dim = batch_codes.shape[1]
    print(f'Latent dimension shape: {latent_dim}')
    
    # Building graph
    w_vec = tf.placeholder('float32', [None, latent_dim], name='w_codes')
    print(f'W in tensorflow graph: {w_vec.shape}')
    encoder_w_tile = tf.tile(w_vec[:, np.newaxis], [1, num_layers, 1])
    print(f'encoder_w_tile size: {encoder_w_tile.shape}')
    reconstructor = Gs.components.synthesis.get_output_for(encoder_w_tile, randomize_noise=False)
    sess = tf.get_default_session()

    save_dir = args.output_dir or './outputs/rebuild_encodings'
    os.makedirs(save_dir, exist_ok=True)

    print('Creating Images...')
    samples = sess.run(reconstructor, {w_vec: batch_codes})
    samples = samples.transpose(0, 2, 3, 1)
    print(f'shape of output: {samples.shape}')
    imwrite(immerge(samples, 4, args.batch_size), '%s/decode_00000_new1.png' % (save_dir))



if __name__ == "__main__":
    main()