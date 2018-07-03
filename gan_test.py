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

# gan_data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
# gan_model = gan.Gan(gan_data)

# for i in range(1,7):
# 	gan_data = data.Data(cat_dim, code_con_dim, 10*i, channel, path, name, split_name, batch_size)
# 	gan_model = gan.Gan(gan_data)
# 	result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(i), 'result')
# 	ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(i), 'weight')
# 	log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(i), 'weight')

# 	if not os.path.isdir(result_dir):
# 		os.makedirs(result_dir)
# 	if not os.path.isdir(ckpt_dir):
# 		os.makedirs(ckpt_dir)
# 	if not os.path.isdir(log_dir):
# 		os.makedirs(log_dir)
# 	gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 2005)
# 	del gan_data
# 	del gan_model

# result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(7), 'result')
# ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(7), 'weight')
# log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(7), 'weight')
# if not os.path.isdir(result_dir):
# 	print('dsfgnkdfsnbdfksbndklsbndklsbnxdklb nkldfsbnjdklsb n', result_dir)
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)
# gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# gan_model = gan.Gan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 2005, G_update_ratio=5, D_update_ratio=1)
# del gan_data
# del gan_model

# result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(8), 'result')
# ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(8), 'weight')
# log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(8), 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)
# gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# gan_model = gan.Gan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 2005, G_update_ratio=1, D_update_ratio=5)
# del gan_data
# del gan_model


# result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(9), 'result')
# ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(9), 'weight')
# log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(9), 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)
# gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# gan_model = gan.Gan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 6005, G_update_ratio=1, D_update_ratio=1)
# del gan_data
# del gan_model


# result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(10), 'result')
# ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(10), 'weight')
# log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(10), 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)
# gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# gan_model = gan.Gan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 1205, G_update_ratio=5, D_update_ratio=5)
# del gan_data
# del gan_model

result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(13), 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(13), 'weight')
log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(13), 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 3005, G_update_ratio=3, D_update_ratio=1)
del gan_data
del gan_model


result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(14), 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(14), 'weight')
log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(14), 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 2005, G_update_ratio=5, D_update_ratio=1)
del gan_data
del gan_model

# result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(12), 'result')
# ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(12), 'weight')
# log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', str(12), 'weight')
# if not os.path.isdir(result_dir):
# 	os.makedirs(result_dir)
# if not os.path.isdir(ckpt_dir):
# 	os.makedirs(ckpt_dir)
# if not os.path.isdir(log_dir):
# 	os.makedirs(log_dir)
# gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# gan_model = gan.Gan(gan_data)
# gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 1205, G_update_ratio=5, D_update_ratio=5)
# del gan_data
# del gan_model