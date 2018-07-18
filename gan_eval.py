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

# Total noise dimension test
for d in range(1, 5):
		result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'noise_dim_test', str(d*10),'result')
		ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'noise_dim_test', str(d*10),'weight')
		log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'noise_dim_test', str(d*10),'weight')
		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)
		if not os.path.isdir(ckpt_dir):
			os.makedirs(ckpt_dir)
		if not os.path.isdir(log_dir):
			os.makedirs(log_dir)
		gan_data = data.Data(cat_dim, code_con_dim, 10*d, channel, path, name, split_name, batch_size)
		gan_data.mutual_penalty_weight = 1
		gan_data.learning_rate = 100
		gan_model = infogan.Info_gan(gan_data)
		gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 10001, G_update_num=5, D_update_num=1, Q_update_num=2)
		del gan_data
		del gan_model

#mutual information weight test
for m in range(1, 5):
		result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'mutual_inform_weight_test', str(m),'result')
		ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'mutual_inform_weight_test', str(m),'weight')
		log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'mutual_inform_weight_test', str(m),'weight')
		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)
		if not os.path.isdir(ckpt_dir):
			os.makedirs(ckpt_dir)
		if not os.path.isdir(log_dir):
			os.makedirs(log_dir)
		gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
		gan_data.mutual_penalty_weight = m
		gan_data.learning_rate = 100
		gan_model = infogan.Info_gan(gan_data)
		gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 10001, G_update_num=5, D_update_num=1, Q_update_num=2)
		del gan_data
		del gan_model

#learning rate test
for r in range(1, 4):
		result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'learning_rate_test', str(10**r) ,'result')
		ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'learning_rate_test', str(10**r), 'weight')
		log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'learning_rate_test', str(10**r), 'weight')
		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)
		if not os.path.isdir(ckpt_dir):
			os.makedirs(ckpt_dir)
		if not os.path.isdir(log_dir):
			os.makedirs(log_dir)
		gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
		gan_data.mutual_penalty_weight = 1
		gan_data.learning_rate = 10**r
		gan_model = infogan.Info_gan(gan_data)
		gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 10001, G_update_num=5, D_update_num=1, Q_update_num=2)
		del gan_data
		del gan_model

#num of Q update
for q in range(1, 4):
		result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'Q_network_update_number_test',str(q) ,'result')
		ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'Q_network_update_number_test',str(q), 'weight')
		log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'Q_network_update_number_test',str(q), 'weight')
		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)
		if not os.path.isdir(ckpt_dir):
			os.makedirs(ckpt_dir)
		if not os.path.isdir(log_dir):
			os.makedirs(log_dir)
		gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
		gan_data.mutual_penalty_weight = 1
		gan_data.learning_rate = 100
		gan_model = infogan.Info_gan(gan_data)
		gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=q)
		del gan_data
		del gan_model

#iteration test
result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'iteration_test', 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'iteration_test', 'weight')
log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'iteration_test', 'weight')
if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(ckpt_dir):
	os.makedirs(ckpt_dir)
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)
gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
gan_data.mutual_penalty_weight = 1
gan_data.learning_rate = 100
gan_model = infogan.Info_gan(gan_data)
gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=q)
del gan_data
del gan_model





# for i in range(1, 8):
# 	for j in range(1, 4):
# 		result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'weight_'+str(i)+'_learning_rate_'+str(j), 'result')
# 		ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'weight_'+str(i)+'_learning_rate_'+str(j), 'weight')
# 		log_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/infogan_exp', 'weight_'+str(i)+'_learning_rate_'+str(j), 'weight')
# 		if not os.path.isdir(result_dir):
# 			os.makedirs(result_dir)
# 		if not os.path.isdir(ckpt_dir):
# 			os.makedirs(ckpt_dir)
# 		if not os.path.isdir(log_dir):
# 			os.makedirs(log_dir)
# 		gan_data = data.Data(cat_dim, code_con_dim, 30, channel, path, name, split_name, batch_size)
# 		gan_data.mutual_penalty_weight = i
# 		gan_data.learning_rate = 10**j
# 		gan_model = infogan.Info_gan(gan_data)
# 		gan_model.train(result_dir, ckpt_dir, log_dir, training_iteration = 100005, G_update_num=5, D_update_num=1, Q_update_num=2)
# 		del gan_data
# 		del gan_model


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

