#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: cnn_mnist.py
Author: zlamberty
Created: 2018-07-09

Description:
    following this tutorial: https://www.tensorflow.org/tutorials/layers

Usage:
    <usage>

"""

import numpy as np
import tensorflow as tf


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

tf.logging.set_verbosity(tf.logging.INFO)


# ----------------------------- #
#   main model definition       #
# ----------------------------- #

def cnn_model_fn(features, labels, mode):
    """Model function for CNN.

    in tensorflow estimators, the `model_fn` completely defines the tensorflow
    model. it must receive at least the three arguments provided here. for more
    details about the arguments (and possible other arguments) see the
    `tf.estimator.Estimator` docstring

    args:
        features (tf.Tensor of dict of name: tf.Tensor kvps): this is the first
            item assumed to be returned by the `input_fn` model defining data
            ingestion. it is either a single feature (as a `tf.Tensor`) or a
            dictionary with keys being feature names and values being feature
            tensors
        labels (tensor-like): the second item assumed to be returned by the
            `input_fn` model defining data ingestion. it is a single output /
            prediction feature (a label)
        mode (str): one of the tf.estimator.ModeKeys values (`EVAL`, `PREDICT`,
            and `TRAIN`).

    """
    # input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # convolutional layer #2 and pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # dense Layer
    pool2_flat = tf.reshape(
        pool2,
        # -1 in first element means figure it out (in the end, batch size)
        # second element has 64 for the 3rd dimension, the number of channels
        # (determined by the number of filters in the 2nd conv layer)
        # the 7s come from the starting height and width (28 each) being
        #   1. left the same (28) in the conv1 because of `padding="same"`
        #   2. halved (14) in the pool1 layer due to `stride=2` and no padding
        #   3. left the same (14) in the conv1 because of `padding="same"`
        #   4. halved (7) in the pool1 layer due to `stride=2` and no padding
        [-1, 7 * 7 * 64]
    )
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        # training flag allows the dropout layer to only apply itself during
        # training runs and not during testing runs
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # logits layer (make 10 predictions)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # short circuit before calculating loss (expensive)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# ----------------------------- #
#   main function               #
# ----------------------------- #

def main(unused_argv):
    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # now create an estimator implementing the model function above
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='/home/zlamberty/tmp/mnist_convnet_model'
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



# ----------------------------- #
#   Command line                #
# ----------------------------- #

if __name__ == '__main__':
    tf.app.run()
