# the file Ir_utils.py and dataset is from www.coursera.org
# Andrew Ng, Co-founder, Coursera; Adjunct Professor, Stanford University;
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lr_utils import load_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loading the dataset (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# train_set_x_orig.shape == (209, 64, 64, 3)
# train_set_y.shape == (1, 209)
# test_set_x_orig.shape == (50, 64, 64, 3)
# test_set_y.shape == (1, 50)

# flatten the matrix
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# standardize the data
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# now we begin to construct the Deep Neural Networks using TensorFlow
# Create placeholders
w1 = tf.placeholder(tf.float32, shape=(5,train_set_x_flatten.shape[0]))
w2 = tf.placeholder(tf.float32, shape=(5,5))
w3 = tf.placeholder(tf.float32, shape=(1,5))
b1 = tf.placeholder(tf.float32, shape=(5, 1))
b2 = tf.placeholder(tf.float32, shape=(5, 1))
b3 = tf.placeholder(tf.float32, shape=(1, 1))
x  = tf.placeholder(tf.float32, shape=(train_set_x_flatten.shape[0],train_set_x_flatten.shape[1]))
y  = tf.placeholder(tf.float32, shape=(train_set_y.shape[0], train_set_y.shape[1]))

# forward propagation
after1 = tf.maximum(tf.matmul(w1, x)+b1,0)
after2 = tf.maximum(tf.matmul(w2, after1)+b2,0)
after3 = tf.tanh(tf.matmul(w3, after2)+b3)
diff  = y - after3

# compute loss
loss = tf.losses.mean_squared_error(after3, y)
grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3 = tf.gradients(loss, [w1, w2, w3, b1, b2, b3])

# run network
with tf.Session() as sess:
    values = {
        x: train_set_x,
        y: train_set_y,
        w1: np.random.randn(5,train_set_x_flatten.shape[0]),
        w2: np.random.randn(5,5),
        w3: np.random.randn(1,5),
        b1: np.zeros((5, 1)),
        b2: np.zeros((5, 1)),
        b3: np.zeros((1, 1))
    }
    learning_rate = 0.5
    for l in range(10):
        out = sess.run([loss, grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3], feed_dict=values)
        loss_temp, grad_w1_temp, grad_w2_temp, grad_w3_temp, grad_b1_temp, grad_b2_temp, grad_b3_temp = out
        values[w1] -= learning_rate * grad_w1_temp
        values[w2] -= learning_rate * grad_w2_temp
        values[w3] -= learning_rate * grad_w3_temp
        values[b1] -= learning_rate * grad_b1_temp
        values[b2] -= learning_rate * grad_b2_temp
        values[b3] -= learning_rate * grad_b3_temp
        if l % 100 == 0:
            print("now: "+str(l))
    after1_test = tf.maximum(tf.matmul(values[w1], test_set_x)+values[b1], 0)
    after2_test = tf.maximum(tf.matmul(values[w2], after1_test)+values[b2], 0)
    after3_test = tf.tanh(tf.matmul(values[w3], after2_test)+values[b3])
    print(after3_test.eval())
    after3_test = tf.maximum(after3_test, 0)

    print(test_set_y)

    print(after3_test.eval())
    index = 5
    plt.imshow(test_set_x[:,index].reshape((64, 64, 3)))
    print("OK")
