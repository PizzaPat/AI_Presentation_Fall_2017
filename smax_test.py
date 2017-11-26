import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


print("x= ",x)
print("W= ",W)
print("b= ",b)

y = tf.nn.softmax(tf.matmul(x, W) + b)

print("y = ",y)

y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

print("cross_entropy = ",cross_entropy)


#Session starts
sess = tf.InteractiveSession()
#Pat: There was no error, only warning when InteractiveSession

tf.global_variables_initializer().run()
