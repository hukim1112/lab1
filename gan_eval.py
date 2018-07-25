from model import data
from model.mnist import megan
import os
import tensorflow as tf

cat_dim = 10
code_con_dim = 3
total_con_dim = 60
channel = 1
path = '/home/dan/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 128

#iteration test
result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/megan_exp', 'test1', 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/megan_exp', 'test1', 'weight')
log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/megan_exp', 'test1', 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 10, channel, path, name, split_name, batch_size)
gan_data.visual_prior_path = '/home/dan/prj/datasets/visual_prior_samples_multinumber'
gan_model = megan.Megan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 30005, G_update_num=5, D_update_num=1, Q_update_num=2)
#gan_model.test()
del gan_data
del gan_model