from model import data
from model.mnist import gan
import os
import tensorflow as tf

cat_dim = 10
code_con_dim = 2
total_con_dim = 60
channel = 1
path = '/home/dan/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 128

gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(gan_data)
result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(12), 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(12), 'weight')
log_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(12), 'log_dir')
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 1001, G_update_ratio = 2)