#Georges Bélanger Albarrán - Handwriting Synthesis

#Based on the following blogs:
#Otoro's Handwriting Generation http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/
#Sam Greydanus Realistic Handwriting https://greydanus.github.io/2016/08/21/handwriting/

#--------------------------------------

# 1.-Import dependencies and other files
import argparse
import time
import os
import numpy as np
import tensorflow as tf

from model import Model
from dataload import *

import ipdb

# 2.- Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size before each gradient step')
parser.add_argument('--num_batches', type=int, default=300, help='number of batches per epoch')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--dropout_prob', type=float, default=0.80, help='prob of keeping activations during dropout')
parser.add_argument('--optimizer', type=str, default='rmsprop', help="Optimizer: 'rmsprop' 'adam'")
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=1.0, help='decay rate for learning rate')
parser.add_argument('--decay_rmsprop', type=float, default=0.95, help='decay rate for rmsprop')
parser.add_argument('--momentum_rmsprop', type=float, default=0.9, help='momentum for rmsprop')
parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
parser.add_argument('--data_scale', type=int, default=50, help='amount to scale data down before training')
parser.add_argument('--rec_dir', type=str, default='./records/', help='location, relative to execution, of log files')
parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
parser.add_argument('--save_path', type=str, default='saved_models/model.ckpt', help='location to save model')
parser.add_argument('--save_every', type=int, default=100, help='number of batches between each save')
parser.add_argument('--vocabulary', type=str, default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', \
						help='default is a-z, A-Z, space, and <UNK> tag')
parser.add_argument('--backprop_steps', type=int, default=150, help='RNN time steps (for backprop)')	
parser.add_argument('--steps_per_char', type=int, default=25, help='expected number of pen points per character')
parser.add_argument('--hidden_size', type=int, default=200, help='size of RNN hidden state')
parser.add_argument('--num_mix_out', type=int, default=8, help='number of gaussian mixtures')
parser.add_argument('--num_mix_window', type=int, default=1, help='number of gaussian mixtures for character window')
parser.add_argument('--train', dest='train', action='store_true', help='train the model')
parser.add_argument('--sample', dest='train', action='store_false', help='sample from the model')
parser.set_defaults(train=True)

args = parser.parse_args()

# 3.- Create record to keep track

class Record():
    def __init__(self, args):
        self.logf = '{}train_scribe.txt'.format(args.rec_dir) if args.train else '{}sample_scribe.txt'.format(args.rec_dir)
        with open(self.logf, 'w') as f: f.write("Generating sequences with RNN\n     \n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print(s)
        with open(self.logf, 'a') as f:
            f.write(s + "\n")


record = Record(args) 
record.write("\nTRAINING")
record.write("{}\n".format(args))
record.write("loading data...")
data_loader = DataLoader(args, record=record)

record.write("building model...")
model = Model(args, record=record)

record.write("attempt to load saved model...")
load_was_success, global_step = model.try_load_model(args.save_path)

v_x, v_y, v_s, v_c = data_loader.validation_data()
valid_inputs = {model.input_data: v_x, model.target_data: v_y, model.char_seq: v_c}

record.write("training...")
model.sess.run(tf.assign(model.decay_rmsprop, args.decay_rmsprop))
model.sess.run(tf.assign(model.momentum_rmsprop, args.momentum_rmsprop ))
running_average = 0.0 ; remember_rate = 0.99
for e in range(global_step/args.num_batches, args.num_epochs):
	model.sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.lr_decay ** e)))
	record.write("learning rate: {}".format(model.learning_rate.eval()))

	c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
	h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
	kappa = np.zeros((args.batch_size, args.num_mix_window, 1))

	for b in range(global_step%args.num_batches, args.num_batches):
		
		i = e * args.num_batches + b
		if global_step is not 0 : i+=1 ; global_step = 0

		if i % args.save_every == 0 and (i > 0):
			model.saver.save(model.sess, args.save_path, global_step = i) ; record.write('SAVED MODEL')

		start = time.time()
		x, y, s, c = data_loader.next_batch()

		feed = {model.input_data: x, model.target_data: y, model.char_seq: c, model.init_kappa: kappa, \
				model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
				model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}

		[train_loss, _] = model.sess.run([model.cost, model.train_op], feed)
		feed.update(valid_inputs)
		feed[model.init_kappa] = np.zeros((args.batch_size, args.num_mix_window, 1))
		[valid_loss] = model.sess.run([model.cost], feed)
		
		running_average = running_average*remember_rate + train_loss*(1-remember_rate)

		end = time.time()
		if i % 10 is 0: record.write("{}/{}, loss = {:.3f}, regloss = {:.5f}, valid_loss = {:.3f}, time = {:.3f}" \
			.format(i, args.num_epochs * args.num_batches, train_loss, running_average, valid_loss, end - start) )