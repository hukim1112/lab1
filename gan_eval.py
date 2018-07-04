from model import data
from model.mnist import infogan
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


# for i in range(1,7):
# 	gan_data = data.Data(cat_dim, code_con_dim, 10*i, channel, path, name, split_name, batch_size)
# 	gan_model = infogan.Info_gan(gan_data)
# 	result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', str(i), 'result')
# 	ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', str(i), 'weight')
# 	log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', str(i), 'weight')
# 	if not os.path.isdir(result_dir):
# 		os.makedirs(result_dir)
# 	if not os.path.isdir(ckpt_dir):
# 		os.makedirs(ckpt_dir)
# 	if not os.path.isdir(log_dir):
# 		os.makedirs(log_dir)
# 	gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 5005, G_update_ratio=5, D_update_ratio=1, Q_update_ratio=1)
# 	del gan_data
# 	del gan_model

result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', str(8), 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', str(8), 'weight')
log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', str(8), 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
gan_model = infogan.Info_gan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=3)
del gan_data
del gan_model