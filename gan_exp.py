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

data = data.Data(cat_dim, code_con_dim, total_con_dim, channel, path, name, split_name, batch_size)
gan_model = gan.Gan(data)



result_dir = '/media/dan/DATA/hukim/Research/srnet/gan_exp/1/result'
ckpt_dir = '/media/dan/DATA/hukim/Research/srnet/gan_exp/1/weight'
gan_model.train(result_dir, ckpt_dir, training_iteration = 1000000)

