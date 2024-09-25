"""
Author: Oren Kraus (https://github.com/okraus, 2013)
Edited by: Myra Paz Masinas (Andrews and Boone Lab, 2023)
"""

import morphologyClassTF as dataClass
import preprocess_images as procIm
import tensorflow as tf
import numpy as np
import nn_layers
import argparse
import os

MAX_STEPS = 10000
WORKERS = 6
SAVE_INTERVAL = 500
NUM_CLASSES = 9
NUM_CHANNELS = 3
CHANNELS2USE=[0,2,4]


parser = argparse.ArgumentParser(description='Train cell cycle model')
parser.add_argument("-i", "--inf_func", help="Inference function for model. "
                                             "Options are: inference_resnet, inference_leo, inference_oren")
parser.add_argument("-l", "--logdir", help="Directory where to store results")
parser.add_argument("-t", "--train_set", help="Path to training set file")
parser.add_argument("-v", "--valid_set", help="Path to validation set file")
args = parser.parse_args()

checkpoint_dir = args.logdir
inference_func_2use = args.inf_func
trainHdf5 = args.train_set
validHdf5 = args.valid_set

def inference_resnet(input_images,is_training):
    ##############################
    ####    resnet model   #######
    ##############################

    hidden1 = nn_layers.conv_layer(input_images, 7, 7, NUM_CHANNELS, 64, 2, 'conv_1',is_training=is_training)
    netVars = [hidden1]
    with tf.name_scope('bottleStack1'):
        netVars.append(nn_layers.bottleStack(netVars[-1], 3, 64, 64, 256, 'bottleStack1',is_training=is_training))
    with tf.name_scope('bottleStack2'):
        netVars.append(nn_layers.bottleStack(netVars[-1], 6, 256, 128, 512, 'bottleStack2',is_training=is_training))
    with tf.name_scope('bottleStack3'):
        netVars.append(nn_layers.bottleStack(netVars[-1], 3, 512, 256, 1024, 'bottleStack3',is_training=is_training))
    netVars.append(tf.reduce_mean(netVars[-1], reduction_indices=[1, 2], name="avg_pool"))
    fc_1 = nn_layers.nn_layer(netVars[-1], 1024, 1000, 'fc_1', act=tf.nn.relu, is_training=is_training)
    logit = nn_layers.nn_layer(fc_1, 1000, NUM_CLASSES, 'final_layer', act=None, is_training=is_training)

    return logit
    ##############################
    ##############################


def inference_leo(input_images,is_training,keep_prob):

    ##############################
    ####     LEO model     #######
    ##############################

    conv1 = nn_layers.conv_layer(input_images, 3, 3, NUM_CHANNELS, 64, 1, 'conv_1',is_training=is_training)
    conv2 = nn_layers.conv_layer(conv1, 3, 3, 64, 64, 1, 'conv_2', is_training=is_training)
    pool1 = nn_layers.pool2_layer(conv2, 'pool1')
    conv3 = nn_layers.conv_layer(pool1, 3, 3, 64, 128, 1, 'conv_3',is_training=is_training)
    conv4 = nn_layers.conv_layer(conv3, 3, 3, 128, 128, 1, 'conv_4', is_training=is_training)
    pool2 = nn_layers.pool2_layer(conv4, 'pool2')
    conv5 = nn_layers.conv_layer(pool2, 3, 3, 128, 256, 1, 'conv_5',is_training=is_training)
    conv6 = nn_layers.conv_layer(conv5, 3, 3, 256, 256, 1, 'conv_6', is_training=is_training)
    conv7 = nn_layers.conv_layer(conv6, 3, 3, 256, 256, 1, 'conv_7', is_training=is_training)
    conv8 = nn_layers.conv_layer(conv7, 3, 3, 256, 256, 1, 'conv_8', is_training=is_training)
    pool3 = nn_layers.pool2_layer(conv8, 'pool3')
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 256])
    fc_1 = nn_layers.nn_layer(pool3_flat, 8 * 8 * 256, 512, 'fc_1', act=tf.nn.relu, is_training=is_training)
    fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
    fc_2 = nn_layers.nn_layer(fc_1_drop, 512, 512, 'fc_2', act=tf.nn.relu,is_training=is_training)
    fc_2_drop = tf.nn.dropout(fc_2, keep_prob)
    logit = nn_layers.nn_layer(fc_2_drop, 512, NUM_CLASSES, 'final_layer', act=None, is_training=is_training)

    return logit
    ##############################
    ##############################


def inference_oren(input_images,is_training,keep_prob):

    ##############################
    ####     Oren model     #######
    ##############################

    conv1 = nn_layers.conv_layer(input_images, 5, 5, NUM_CHANNELS, 32, 1, 'conv_1',is_training=is_training)
    pool2 = nn_layers.pool2_layer(conv1, 'pool1')
    conv2 = nn_layers.conv_layer(pool2, 5, 5, 32, 64, 1, 'conv_2',is_training=is_training)
    pool2 = nn_layers.pool2_layer(conv2, 'pool2')
    conv3 = nn_layers.conv_layer(pool2, 5, 5, 64, 64, 1, 'conv_3',is_training=is_training)
    pool3 = nn_layers.pool2_layer(conv3, 'pool3')

    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])
    pool3_drop = tf.nn.dropout(pool3_flat, keep_prob)
    fc_1 = nn_layers.nn_layer(pool3_drop, 8 * 8 * 64, 1024, 'fc_1', act=tf.nn.relu, is_training=is_training)
    fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
    logit = nn_layers.nn_layer(fc_1_drop, 1024, NUM_CLASSES, 'final_layer', act=None,  is_training=is_training)

    return logit
    ##############################
    ##############################

def loss(predicted_y,labeled_y):
    with tf.name_scope('cross_entropy'):
        diff = labeled_y * tf.log(tf.clip_by_value(predicted_y,1e-16,1.0))
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.summary.scalar('cross entropy', cross_entropy)

    return cross_entropy

def loss_logits(logits,labeled_y,baseName):
    with tf.name_scope('cross_entropy'):
        logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labeled_y,
                                                                  name='sigmoid_cross_entropy')
        cross_entropy = tf.reduce_mean(logistic_losses)
        tf.summary.scalar(baseName+'_cross entropy', cross_entropy)

    return cross_entropy


def accuracy(predicted_y,labeled_y,baseName):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(labeled_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(baseName+'_accuracy', accuracy)

    return accuracy

def getSpecificChannels(flatImageData,channels,imageSize=64):
    return np.hstack(([flatImageData[:,c*imageSize**2:(c+1)*imageSize**2] for c in channels]))

def processBenBatch(curBatch):
    curImages = getSpecificChannels(curBatch['data'],CHANNELS2USE)
    curLabels = curBatch['Index'][:,:22]
    return {'data':curImages,'Index':curLabels}

def train(inference_function):
    print('\n\n',inference_function,'\n\n')
    sess = tf.Session()
    dequeueSize = 100
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    decay_step = 25
    decay_rate = 0.98
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decay_step, decay_rate, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    cropSize = 60
    batchSize = 128
    stretchLow = 0.1 # stretch channels lower percentile
    stretchHigh = 99.9 # stretch channels upper percentile

    imSize = 64
    numClasses = NUM_CLASSES
    numChan = NUM_CHANNELS
    data = dataClass.Data(trainHdf5,['data','Index'],batchSize)
    dataTest = dataClass.Data(validHdf5,['data','Index'],batchSize * 2) # larger batch size at test time

    ### define model
    is_training = tf.placeholder(tf.bool, [], name='is_training') # for batch norm
    input = tf.placeholder('float32', shape = [None,cropSize,cropSize,numChan], name='input')  # for batch norm
    labels = tf.placeholder('float32', shape = [None,numClasses], name='labels')  # for batch norm
    keep_prob = tf.placeholder(tf.float32)

    if inference_func_2use == "inference_resnet":
        logits = inference_function(input, is_training)
    else:
        logits = inference_function(input, is_training, keep_prob)
    predicted_y = tf.nn.softmax(logits, name='softmax')

    # test graph
    with tf.name_scope('train'):
        train_acc = accuracy(predicted_y,labels,'train')
        cross_entropy = loss_logits(logits, labels,'train')
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
    with tf.name_scope('test'):
        test_acc = accuracy(predicted_y,labels,'test')
        test_loss = loss_logits(logits, labels,'test')

    saver = tf.train.Saver(tf.all_variables(),max_to_keep=100)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoint_dir + '/train',
                                          sess.graph)

    sess.run(tf.initialize_all_variables())

    # training loop
    for i in range(MAX_STEPS):
        print('step ',i)

        if i % 50 == 0:  # Record execution stats

            batch = dataTest.getBatch()
            batch = processBenBatch(batch)
            #pdb.set_trace()
            processedBatch=procIm.preProcessImages(batch['data'],
                                       imSize,cropSize,numChan,
                                       rescale=False,stretch=True,
                                       means=None,stds=None,
                                       stretchLow=stretchLow,stretchHigh=stretchHigh)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, acc, cur_test_loss = sess.run([merged, test_acc, test_loss],

                      feed_dict={is_training: False,
                                 keep_prob:1.0,
                                 input: processedBatch,
                                 labels: batch['Index']},
                        options=run_options,
                        run_metadata=run_metadata)

            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
            lr,g = sess.run([learning_rate,global_step])
            print('Valid accuracy at step %s: %s loss %s learning_rate: %s global step: %s' % (i, acc,cur_test_loss,lr,g))

        else:  # Record a summary
            batch = data.getBatch()
            batch = processBenBatch(batch)
            processedBatch=procIm.preProcessImages(batch['data'],
                                       imSize,cropSize,numChan,
                                       rescale=False,stretch=True,
                                       means=None,stds=None,
                                       stretchLow=stretchLow,stretchHigh=stretchHigh)

            summary, _ , acc = sess.run([merged, train_step, train_acc],
                                           feed_dict={is_training: True,
                                                      keep_prob:0.5,
                                                      input: processedBatch,
                                                      labels: batch['Index']})
            train_writer.add_summary(summary, i)
            print('Train cccuracy at step %s: %s ' % (i, acc))

        if i % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, inference_func_2use+'_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)

    train_writer.close()

def main(_):
    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)
    tf.gfile.MakeDirs(checkpoint_dir)

    if inference_func_2use=="inference_leo":
        inference_func = inference_leo
    elif inference_func_2use=="inference_resnet":
        inference_func = inference_resnet
    elif inference_func_2use=="inference_oren":
        inference_func = inference_oren
    else:
        raise NameError("inference func must be [inference_resnet,inference_leo]")
    train(inference_func)


if __name__ == '__main__':
    tf.app.run()
