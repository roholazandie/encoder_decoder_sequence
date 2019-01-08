import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from nmt.train import build_train_model
from nmt.model import train_infer
from nmt.evaluation import build_eval_model
from nmt.iterator_util import get_iterator, get_tables, get_datasets
from nmt.hyperparameters import get_hyperparameters

hyperparameters = get_hyperparameters()

train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():
    source_vocab_table, target_vocab_table = get_tables(hyperparameters["source_vocab_file"],
                                                        hyperparameters["target_vocab_file"])

    source_dataset, target_dataset = get_datasets(hyperparameters["train_source_file"],
                                                  hyperparameters["train_target_file"])

    train_iterator = get_iterator(hyper_parameters=hyperparameters,
                                  source_vocab_table=source_vocab_table,
                                  target_vocab_table=target_vocab_table,
                                  source_dataset=source_dataset,
                                  target_dataset=target_dataset)

    mode = tf.contrib.learn.ModeKeys.TRAIN

    loss, logits, sample_id = build_train_model(train_graph,
                                                train_iterator,
                                                source_vocab_table,
                                                target_vocab_table,
                                                hyperparameters,
                                                mode)

    global_step = tf.Variable(0, trainable=False)
    train_op = train_infer(mode=mode,
                           iterator=train_iterator,
                           hyper_parameters=hyperparameters,
                           loss=loss,
                           logits=logits)

with eval_graph.as_default():
    eval_iterator = None
    eval_model = build_eval_model()

with tf.Session(graph=train_graph) as train_sess:
    # initializeing global vars, tables and iterator
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.iterator.initializer)

    _, loss_value = train_sess.run([train_op, loss])  # add other things
    print(loss_value)

# with tf.Session(graph=eval_graph) as eval_sess:
#     pass
#
#
# with tf.Session(graph=infer_graph) as infer_sess:
#     pass
