from model import data
from model.mnist import gan
import os
import tensorflow as tf

cat_dim = 10
code_con_dim = 2
total_con_dim = 30
channel = 1
path = '/home/dan/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 32

gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(gan_data)

result_dir = '/home/dan/prj/results/tmp'
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(6), 'weight')
gan_model.evaluate_with_random_sample(result_dir, ckpt_dir)