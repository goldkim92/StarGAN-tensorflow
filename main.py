#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:18:55 2017

@author: jm
"""

import argparse
import os
import tensorflow as tf
from model import stargan

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu_number', type=str, default='0')
parser.add_argument('--data_dir', type=str, default=os.path.join('.','data','celebA'))
parser.add_argument('--log_dir', type=str, default=os.path.join('.','assets','log'))
parser.add_argument('--ckpt_dir', type=str, default=os.path.join('.','assets','checkpoint'))
parser.add_argument('--sample_dir', type=str, default=os.path.join('.','assets','sample'))
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--image_channel', type=int, default=3)
parser.add_argument('--nf', type=int, default=64) # number of filters
parser.add_argument('--n_label', type=int, default=10)
parser.add_argument('--lambda_cls', type=int, default=1)
parser.add_argument('--lambda_rec', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0001) # learning_rate
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--snapshot', type=int, default=100)
parser.add_argument('--adv_loss', type=str, default='GAN', help='GAN or WGAN')

args = parser.parse_args()

def main(_):
    # setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
    tf.reset_default_graph()

    # make directory if not exist
    try: os.makedirs(args.log_dir)
    except: pass
    try: os.makedirs(args.ckpt_dir)
    except: pass
    try: os.makedirs(args.sample_dir)
    except: pass

    # run session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = stargan(sess,args)
        model.train()
#        if args.phase == 'train' else model.test()

# run main function
if __name__ == '__main__':
    tf.app.run()