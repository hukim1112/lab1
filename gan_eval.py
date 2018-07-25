from model import data
from model.mnist import megan1, megan2, megan3
import os
import tensorflow as tf

cat_dim = 10
code_con_dim = 3
total_con_dim = 60
channel = 1
path = '/home/hukim/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 128

#megan optimizer test
result_dir = os.path.join('/home/hukim/prj/results', 'test1', 'result')
ckpt_dir =  os.path.join('/home/hukim/prj/results', 'test1', 'weight')
log_dir =  os.path.join('/home/hukim/prj/results', 'test1', 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 10, channel, path, name, split_name, batch_size)
gan_data.visual_prior_path = '/home/hukim/prj/datasets/visual_prior_samples_multinumber'
gan_model = megan1.Megan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=2)
#gan_model.test()
del gan_data
del gan_model





#megan optimizer test
result_dir = os.path.join('/home/hukim/prj/results', 'test2', 'result')
ckpt_dir =  os.path.join('/home/hukim/prj/results', 'test2', 'weight')
log_dir =  os.path.join('/home/hukim/prj/results', 'test2', 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 10, channel, path, name, split_name, batch_size)
gan_data.visual_prior_path = '/home/hukim/prj/datasets/visual_prior_samples_multinumber'
gan_model = megan2.Megan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=2)
#gan_model.test()
del gan_data
del gan_model






#megan optimizer test
result_dir = os.path.join('/home/hukim/prj/results', 'test3', 'result')
ckpt_dir =  os.path.join('/home/hukim/prj/results', 'test3', 'weight')
log_dir =  os.path.join('/home/hukim/prj/results', 'test3', 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 10, channel, path, name, split_name, batch_size)
gan_data.visual_prior_path = '/home/hukim/prj/datasets/visual_prior_samples_multinumber'
gan_model = megan3.Megan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=2)
#gan_model.test()
del gan_data
del gan_model