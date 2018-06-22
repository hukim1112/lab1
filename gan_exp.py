from model import data
from model.mnist import gan
import tensorflow as tf

cat_dim = 10
code_con_dim = 2
total_con_dim = 60
channel = 1
path = '/home/dan/prj/datasets/mnist'
name = 'mnist'
split_name = 'train'
batch_size = 128

for i in range(1, 6):
	data = data.Data(cat_dim, code_con_dim, 10*i, channel, path, name, split_name, batch_size)
	gan_model = gan.Gan(data)
	result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', i, 'result')
	ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', i, 'weight')
	gan_model.train(result_dir, ckpt_dir, training_iteration = 100000, D_update_ratio = 1)
	del gan_model


data = data.Data(cat_dim, code_con_dim, 10*3, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(data)
result_dir = os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', 6, 'result')
ckpt_dir =  os.path.join('/media/dan/DATA/hukim/Research/srnet/gan_exp', 6, 'weight')
gan_model.train(result_dir, ckpt_dir, training_iteration = 100000, D_update_ratio = 5)