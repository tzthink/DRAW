import tensorflow as tf
import numpy as np
import os
from ReadData import ReadData
import time

# tensorflow flags
tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", True, "enable attention fro writer")

FLAGS = tf.flags.FLAGS

# model parameters
A, B = 32, 32
img_size = B * A
enc_size = 256
dec_size = 256
read_n = 5  # read glimpse grid width/height
write_n = 5  # write glimpse grid width/height
read_size = 2 * read_n * read_n if FLAGS.read_attn else 2 * img_size
write_size = write_n * write_n if FLAGS.write_attn else img_size
z_size = 10  # QSampler output size
T = 10  # generation sequence length
batch_size = 64  # training minibatch size
train_iters = 10000
learning_rate = 1e-3  # learning rate for optimizer
eps = 1e-8  # epsilon for numerical stability

# build model
DO_SHARE = None  # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32, shape=(batch_size, img_size))  # input (batch_size * img_size)
e = tf.random_normal((batch_size, z_size), mean=0, stddev=1)  # Qsample noise
# LONG SHORT-TERM MEMORY
lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, state_is_tuple=True)  # encoder Op
lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, state_is_tuple=True)  # decoder Op


def linear(x, output_dim):
    """
    affine transformation Wx + b
    assumes x.shape = (batch_size, num_features)
    """
    w = tf.get_variable("w", [x.get_shape()[1], output_dim])
    b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w) + b


def filterbank(gx, gy, sigma2, delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta  # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta  # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2 * sigma2)))
    Fy = tf.exp(-tf.square((b - mu_y) / (2 * sigma2)))
    # normalize, sum over A and B dims
    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
    return Fx, Fy


def attn_window(scope, h_dec, N):
    with tf.variable_scope(scope, reuse=DO_SHARE):
        params = linear(h_dec, 5)
    gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(1, 5, params)
    gx = (A + 1) / 2 * (gx_ + 1)
    gy = (B + 1) / 2 * (gy_ + 1)
    sigma2 = tf.exp(log_sigma2)
    delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)  # batch * N
    return filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma),)


# read
def read_no_attn(x, x_hat, h_dec_prev):
    return tf.concat(1, [x, x_hat])


def read_attn(x, x_hat, h_dec_prev):
    Fx, Fy, gamma = attn_window("read", h_dec_prev, read_n)

    def filter_img(img, Fx, Fy, gamma, N):
        Fxt = tf.transpose(Fx, perm=[0, 2, 1])
        img = tf.reshape(img, [-1, B, A])
        glimpse = tf.batch_matmul(Fy, tf.batch_matmul(img, Fxt))
        glimpse = tf.reshape(glimpse, [-1, N * N])
        return glimpse * tf.reshape(gamma, [-1, 1])

    x = filter_img(x, Fx, Fy, gamma, read_n)  # batch x (read_n * read_n)
    x_hat = filter_img(x_hat, Fx, Fy, gamma, read_n)
    return tf.concat(1, [x, x_hat])  # concat along feature axis


read = read_attn if FLAGS.read_attn else read_no_attn


# encode
def encode(state, input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read, h_dec_prev)
    returns : (output, new_state)
    """
    with tf.variable_scope("encoder", reuse=DO_SHARE):
        return lstm_enc(input, state)


# Q-Sampler (variational autoencoder)
def sampleQ(h_enc):
    """
    samples Zt ~ normrnd(mu, sigma) via reparameterization trick for normal dist
    mu is (batch, z_size)
    """
    with tf.variable_scope("mu", reuse=DO_SHARE):
        mu = linear(h_enc, z_size)
    with tf.variable_scope("sigma", reuse=DO_SHARE):
        logsigma = linear(h_enc, z_size)
        sigma = tf.exp(logsigma)
    return (mu + sigma * e, mu, logsigma, sigma)


# decoder
def decode(state, input):
    with tf.variable_scope("decoder", reuse=DO_SHARE):
        return lstm_dec(input, state)


# writer
def write_no_attn(h_dec):
    with tf.variable_scope("write", reuse=DO_SHARE):
        return linear(h_dec, img_size)


def write_attn(h_dec):
    with tf.variable_scope("writeW", reuse=DO_SHARE):
        w = linear(h_dec, write_size)  # batch x (write_n * write_n)
    N = write_n
    w = tf.reshape(w, [batch_size, N, N])
    Fx, Fy, gamma = attn_window("write", h_dec, write_n)
    Fyt = tf.transpose(Fy, perm=[0, 2, 1])
    wr = tf.batch_matmul(Fyt, tf.batch_matmul(w, Fx))
    wr = tf.reshape(wr, [batch_size, B * A])
    return wr * tf.reshape(1.0 / gamma, [-1, 1])


write = write_attn if FLAGS.write_attn else write_no_attn

# state variables
cs = [0] * T  # sequence of canvases
# gaussian params generated by SampleQ.
# We will need these for computing loss.
mus, logsigmas, sigmas = [0] * T, [0] * T, [0] * T
# initial states
h_dec_prev = tf.zeros((batch_size, dec_size))
enc_state = lstm_enc.zero_state(batch_size, tf.float32)
dec_state = lstm_dec.zero_state(batch_size, tf.float32)

################
# DRAW MODEL
################

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t - 1]
    x_hat = x - tf.sigmoid(c_prev)
    r = read(x, x_hat, h_dec_prev)
    h_enc, enc_state = encode(enc_state, tf.concat(1, [r, h_dec_prev]))
    z, mus[t], logsigmas[t], sigmas[t] = sampleQ(h_enc)
    h_dec, dec_state = decode(dec_state, z)
    cs[t] = c_prev + write(h_dec)  # store results
    h_dec_prev = h_dec
    DO_SHARE = True  # from now on, share variables


# loss function
def binary_crossentropy(t, o):
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


# reconstruction term appears to have been collapsed down to a single
# scalar value (rather than one per item in minibatch)
x_recons = tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums
# across minibatches
Lx = tf.reduce_sum(binary_crossentropy(x, x_recons), 1)  # reconstruction term
Lx = tf.reduce_mean(Lx)

kl_terms = [0] * T
for t in range(T):
    mu2 = tf.square(mus[t])
    sigma2 = tf.square(sigmas[t])
    logsigma = logsigmas[t]
    # each kl term is (1 * minibatch)
    kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - T * 0.5
KL = tf.add_n(kl_terms)  # this is 1 * minibatch, corresponding to summing kl_terms from 1:T
Lz = tf.reduce_mean(KL)

cost = Lx + Lz

# optimizer
########################
# Gradient clipping needs to happen after computing the gradients,
# but before applying them to update the model's parameters.
########################
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads = optimizer.compute_gradients(cost)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
train_op = optimizer.apply_gradients(grads)

# run training
rd = ReadData()

fetches = []
fetches.extend([Lx, Lz, train_op])
Lxs = [0] * train_iters
Lzs = [0] * train_iters

sess = tf.InteractiveSession()

saver = tf.train.Saver()  # saves (variables) learned during training
tf.initialize_all_variables().run()
# saver.restore(sess, "/tmp/draw/drawmodel.ckpt")   # to restore from model, uncomment this line

start = time.time()
for i in range(train_iters):
    xtrain = rd.get_next_batch(batch_size=batch_size)  # xtrain is (batch_size x img_size)
    feed_dict = {x: xtrain}
    results = sess.run(fetches, feed_dict)
    Lxs[i], Lzs[i], _ = results
    if i % 100 == 0:
        print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))
end = time.time()
duration = end - start
print("The time used is: %f s" % duration)

# training finished
print(feed_dict)
print(type(feed_dict))
canvases = sess.run(cs, feed_dict)  # generate some examples
canvases = np.array(canvases)  # T x batch x img_size

out_file = os.path.join(FLAGS.data_dir, "draw_data.npy")
np.save(out_file, [canvases, Lxs, Lzs])
print("Outputs saved in file: %s" % out_file)

ckpt_file = os.path.join(FLAGS.data_dir, "drawmodel.ckpt")
print("Model saved in file: %s" % saver.save(sess, ckpt_file))

sess.close()

print ('Done drawing! Have a nice day!')