import tensorflow as tf
import numpy as np
from const import *
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


def _conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 tf.random_normal_initializer(stddev=WEIGHT_INIT))
        return tf.nn.conv2d(x, filter, [1, strides, strides, 1], 'SAME')

def _relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def _FC(name, x, out_dim, keep_rate, activation='relu'):
    assert (activation == 'relu') or (activation == 'softmax') or (activation == 'linear')
    with tf.variable_scope(name):
        dim = x.get_shape().as_list()
        dim = np.prod(dim[1:])
        x = tf.reshape(x, [-1, dim])
        W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT))
        b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, W, b)
        if activation == 'relu':
            x = _relu(x)
        else:
            if activation == 'softmax':
                x = tf.nn.softmax(x)

        if activation != 'relu':
            return x
        else:
            return tf.nn.dropout(x, keep_rate)


def _max_pool(x, filter, stride):
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], 'SAME')


def batch_norm(x, n_out, phase_train=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def VGG_ConvBlock(name, x, in_filters, out_filters, repeat, strides, phase_train):
    with tf.variable_scope(name):
        for layer in range(repeat):
            scope_name = name + '_' + str(layer)
            x = _conv(scope_name, x, 3, in_filters, out_filters, strides)
            if USE_BN:
                x = batch_norm(x, out_filters, phase_train)
            x = _relu(x)

            in_filters = out_filters

        x = _max_pool(x, 2, 2)
        return x

def Input():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])
    y_ = tf.placeholder(tf.float32, [None, 3])
    mask = tf.placeholder(tf.float32, [None])

    return x, y_, mask


def BKNetModel(x):
    phase_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    x = VGG_ConvBlock('Block1', x, 1, 32, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block2', x, 32, 64, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block3', x, 64, 128, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block4', x, 128, 256, 3, 1, phase_train)
    # print(x.get_shape())

    # Smile branch
    smile_fc1 = _FC('smile_fc1', x, 256, keep_prob)
    smile_fc2 = _FC('smile_fc2', smile_fc1, 256, keep_prob)
    y_smile_conv = _FC('smile_softmax', smile_fc2, 2, keep_prob, 'softmax')

    # Gender branch
    # gender_fc1 = _FC('gender_fc1', x, 256, keep_prob)
    # gender_fc2 = _FC('gender_fc2', gender_fc1, 256, keep_prob)
    # y_gender_conv = _FC('gender_softmax', gender_fc2, 2, keep_prob, 'softmax')

    # Age branch
    # age_fc1 = _FC('age_fc1', x, 256, keep_prob)
    # age_fc2 = _FC('age_fc2', age_fc1, 256, keep_prob)
    # y_age_conv = _FC('age_softmax', age_fc2, 4, keep_prob, 'softmax')

    return y_smile_conv, phase_train, keep_prob

def selective_loss(y_face_mask, y_, mask):
    vector_zero = tf.constant(0., tf.float32, [BATCH_SIZE])

    face_mask = tf.cast(tf.equal(mask, 0.), tf.float32) #vector_zero

    tf.add_to_collection('face_mask', face_mask)
    y_face = tf.slice(y_, [0, 0], [-1, 2])
    tf.add_to_collection('y_face', y_face)

    face_mask_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_face * tf.log(y_face_mask), axis=1) * face_mask) / tf.clip_by_value(
        tf.reduce_sum(face_mask), 1, 1e9)
    l2_loss = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            l2_loss.append(tf.nn.l2_loss(var))
    l2_loss = WEIGHT_DECAY * tf.add_n(l2_loss)

    total_loss = face_mask_cross_entropy + l2_loss

    return face_mask_cross_entropy, l2_loss, total_loss


def train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEP, DECAY_LR_RATE, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss,
                                                                                                                   global_step=global_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step