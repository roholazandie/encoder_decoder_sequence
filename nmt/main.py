import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from nmt.model import Model
from nmt.train import Train
from nmt.evaluation import build_eval_model
from nmt.iterator_util import DatasetIterator
from nmt.hyperparameters import get_hyper_parameters
from nmt.nmt_util import get_translation
import nmt.inference as inference
import random

hyper_parameters = get_hyper_parameters()


def create_train_model():
    train_graph = tf.Graph()
    mode = tf.contrib.learn.ModeKeys.TRAIN

    train_model = Model(mode, hyper_parameters)
    dataset_iterator = DatasetIterator(hyper_parameters)
    train = Train(mode, hyper_parameters)

    with train_graph.as_default():


        source_vocab_table, target_vocab_table = dataset_iterator.get_tables(share_vocab=False)

        source_dataset, target_dataset = dataset_iterator.get_datasets()

        train_iterator = dataset_iterator.get_iterator(source_vocab_table=source_vocab_table,
                                                      target_vocab_table=target_vocab_table,
                                                      source_dataset=source_dataset,
                                                      target_dataset=target_dataset,
                                                      source_max_len=hyper_parameters["source_max_len_train"],
                                                      target_max_len = hyper_parameters["target_max_len_train"],
                                                       #skip_count=skip_count_place_holder #todo probably we need this
                                                       )


        logits, loss, final_context_state, sample_id = train_model.build_model(train_iterator, target_vocab_table)

        train.configure_train_eval_infer(iterator=train_iterator,
                                                                 logits=logits,
                                                                 loss=loss,
                                                                 sample_id=sample_id,
                                                                 final_state=final_context_state)

    return train_graph, train_iterator, train



def create_eval_model():
    eval_graph = tf.Graph()

    mode = tf.contrib.learn.ModeKeys.EVAL

    eval_model = Model(mode, hyper_parameters)
    dataset_iterator = DatasetIterator(hyper_parameters)
    eval = Train(mode, hyper_parameters)

    with eval_graph.as_default():
        source_vocab_file = hyper_parameters["source_vocab_file"]
        target_vocab_file = hyper_parameters["target_vocab_file"]

        source_vocab_table, target_vocab_table = dataset_iterator.get_tables(share_vocab=False)

        reverse_target_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(target_vocab_file,
                                                                                       default_value="UNK")

        source_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        target_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        source_dataset = tf.data.TextLineDataset(source_file_placeholder)
        target_dataset = tf.data.TextLineDataset(target_file_placeholder)

        eval_iterator = dataset_iterator.get_iterator(source_vocab_table=source_vocab_table,
                                                     target_vocab_table=target_vocab_table,
                                                     source_dataset=source_dataset,
                                                     target_dataset=target_dataset,
                                                     source_max_len=hyper_parameters["source_max_len_infer"],
                                                     target_max_len = hyper_parameters["target_max_len_infer"]
                                                      )

        logits, loss, final_context_state, sample_id = eval_model.build_model(eval_iterator, target_vocab_table)


        eval.configure_train_eval_infer(iterator=eval_iterator,
                                        logits=logits,
                                        loss=loss,
                                        sample_id=sample_id,
                                        final_state=final_context_state,
                                        reverse_target_vocab_table=reverse_target_vocab_table)


    return eval_graph, eval, eval_iterator, source_file_placeholder, target_file_placeholder


def create_infer_model():
    infer_graph = tf.Graph()
    mode = tf.contrib.learn.ModeKeys.INFER

    infer_model = Model(mode, hyper_parameters)
    dataset_iterator = DatasetIterator(hyper_parameters)
    infer = Train(mode, hyper_parameters)

    with infer_graph.as_default():
        source_vocab_file = hyper_parameters["source_vocab_file"]
        target_vocab_file = hyper_parameters["target_vocab_file"]

        reverse_target_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(target_vocab_file,
                                                                                       default_value="UNK")

        source_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        source_dataset = tf.data.Dataset.from_tensor_slices(source_placeholder)

        source_vocab_table, target_vocab_table = dataset_iterator.get_tables(share_vocab=False)

        infer_iterator = dataset_iterator.get_infer_iterator(source_dataset=source_dataset,
                                                             source_vocab_table=source_vocab_table,
                                                             source_max_len=hyper_parameters["source_max_len_infer"]
                                                            )

        logits, loss, final_context_state, sample_id = infer_model.build_model(infer_iterator, target_vocab_table)

        infer.configure_train_eval_infer(iterator=infer_iterator,
                                         logits=logits,
                                         loss=loss,
                                         sample_id=sample_id,
                                         final_state=final_context_state,
                                         reverse_target_vocab_table=reverse_target_vocab_table)

        return infer_graph, infer, infer_iterator, source_placeholder, batch_size_placeholder




train_graph, train_iterator, train = create_train_model()
eval_graph, eval, eval_iterator, source_file_placeholder, target_file_placeholder = create_eval_model()
infer_graph, infer, infer_iterator, source_placeholder, batch_size_placeholder = create_infer_model()

train_sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)
infer_sess = tf.Session(graph=infer_graph)


# initializeing global vars, tables and iterator
train.initialize_model(train_graph, train_sess)
eval.initialize_model(eval_graph, eval_sess)# do we need to initialize eval session this way?
infer.initialize_model(infer_graph, infer_sess)

global_step_value = 0
while global_step_value < hyper_parameters["num_training_steps"]:

    train_sess.run(train_iterator.iterator.initializer)
    try:
        _, train_loss, predict_count, global_step_value, word_count, grad_norm, learning_rate = train.train(train_sess)
        print(train_loss)
    except tf.errors.OutOfRangeError:
        print("Finished!")

        #internal eval
        eval_perplexity = eval.compute_perplexity("eval", eval_sess, eval_iterator, source_file_placeholder, target_file_placeholder)
        #infer_perplexity = infer.compute_perplexity("infer", infer_sess, infer_iterator, )

        # sample decoding
        sample_source_data = inference.load_data(hyper_parameters["eval_source_file"])
        sample_target_data = inference.load_data(hyper_parameters["eval_target_file"])

        decode_id = random.randint(0, len(sample_source_data) - 1)

        iterator_feed_dict = {source_placeholder: [sample_source_data[decode_id]],
                              batch_size_placeholder: 1}

        #nmt_outputs, attention_summary = infer.decode(infer_sess)
        _, infer_summary, _, nmt_outputs = infer.infer(infer_sess)

        translation = get_translation(nmt_outputs, sent_id=0, target_eos=hyper_parameters["eos"])





# with tf.Session(graph=eval_graph) as eval_sess:
#     pass
#
#
# with tf.Session(graph=infer_graph) as infer_sess:
#     pass
