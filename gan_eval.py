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
for i in range(1, 11):
	for j in range(1, 4):
		result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'weight_'+str(i)+'_learning_rate_'+str(j), 'result')
		ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'weight_'+str(i)+'_learning_rate_'+str(j), 'weight')
		log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'weight_'+str(i)+'_learning_rate_'+str(j), 'weight')
		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)
		if not os.path.isdir(ckpt_dir):
			os.makedirs(ckpt_dir)
		if not os.path.isdir(log_dir):
			os.makedirs(log_dir)
		gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
		gan_data.mutual_penalty_weight = i
		gan_data.learning_rate = 10**j
		gan_model = infogan.Info_gan(gan_data)
		gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 50005, G_update_num=5, D_update_num=1, Q_update_num=3)
		del gan_data
		del gan_model

# for i in range(1, 4):
# 	result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'con_loss_x2', 'result')
# 	ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'con_loss_x2', 'weight')
# 	log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'con_loss_x2', 'weight')
# 	if not os.path.isdir(result_dir):
# 		os.makedirs(result_dir)
# 	if not os.path.isdir(ckpt_dir):
# 		os.makedirs(ckpt_dir)
# 	if not os.path.isdir(log_dir):
# 		os.makedirs(log_dir)
# 	gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# 	gan_data.mutual_penalty_weight = i
# 	gan_data.learning_rate = 10**i
# 	gan_model = infogan.Info_gan(gan_data)
# 	#gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 50005, G_update_num=5, D_update_num=1, Q_update_num=3)
# 	del gan_data
# 	del gan_model