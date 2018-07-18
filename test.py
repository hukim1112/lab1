
import tensorflow as tf
ds = tf.contrib.distributions
from tensorflow.python.ops.losses import losses
# q_cont = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [-1.0, -2.0, -3.0, -4.0]]
# sigma_cont = tf.ones_like(q_cont)
# q_cont = ds.Normal(loc=q_cont, scale=sigma_cont)

#   # q_cat = predicted_distributions[0]
#   # q_cat = ds.Categorical(q_cat)
# sess = tf.Session()

# print(sess.run(tf.reduce_mean(q_cont.log_prob([[1.0, 1.0, 1.0, 1.0], [0.0, -2.0, -3.0, 0.0], [-1.0, -2.0, -3.0, -4.0]]), axis = 0)   ))




q_cat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
q_cat = ds.Categorical(q_cat)
code_cat = tf.argmax([[1, 0, 0], [0, 1, 0], [1, 0, 0]], axis = 1)
log_prob_cat = [tf.reduce_mean(q_cat.log_prob(code_cat))]


q_cont_ = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [-1.0, -2.0, -3.0, -4.0]]
sigma_cont = tf.ones_like(q_cont_)
q_cont = ds.Normal(loc=q_cont_, scale=sigma_cont)
log_prob_con = tf.reduce_mean(q_cont.log_prob(q_cont_), axis = 0)


log_prob = tf.concat([log_prob_cat, log_prob_con], axis=0)
#print('log_prob shape', log_prob.shape)

sess = tf.Session()

loss = -1 * losses.compute_weighted_loss(log_prob, 1)
print(sess.run([log_prob, loss]))