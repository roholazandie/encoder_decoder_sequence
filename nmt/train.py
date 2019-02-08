import tensorflow as tf
import numpy as np


class Train():

    def __init__(self, mode, hyper_parameters):
        self.hyper_parameters = hyper_parameters
        self.mode = mode


    def gradient_clip(self, gradients):
        """Clipping gradients of a model."""
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.hyper_parameters["max_gradient_norm"])
        gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

        return clipped_gradients, gradient_norm_summary, gradient_norm


    def configure_train_eval_infer(self, iterator, logits, loss, sample_id, final_state, reverse_target_vocab_table=None):

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.word_count = tf.reduce_sum(iterator.source_sequence_length) + tf.reduce_sum(iterator.target_sequence_length)

            self.predict_count = tf.reduce_sum(iterator.target_sequence_length)
            self.train_loss = loss

            self.learning_rate = tf.constant(self.hyper_parameters["learning_rate"])#todo replace with _get_learning_rate_warmup and _get_learning_rate_decay

            params = tf.trainable_variables()

            # Optimizer
            if self.hyper_parameters["optimizer"] == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.hyper_parameters["optimizer"] == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)
            else:
                raise ValueError("Unknown Optimizer "+str(self.hyper_parameters["optimizer"]))

            # Gradients
            gradients = tf.gradients(self.train_loss, params)# colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_grads, grad_norm_summary, grad_norm = self.gradient_clip(gradients)

            self.grad_norm_summary = grad_norm_summary
            self.grad_norm = grad_norm

            self.global_step = tf.Variable(0, trainable=False)

            self.train_op = opt.apply_gradients(zip(clipped_grads, params), global_step=self.global_step)

            #todo add train summary here

            #return self.train_op, self.global_step

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = loss

        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits = logits
            self.sample_id = sample_id
            self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(sample_id))
            self.infer_summary = tf.no_op()

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            ## Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(iterator.target_sequence_length)
            #todo add summary



    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN

        return sess.run([self.train_op,
                         self.train_loss,
                         self.predict_count,
                         self.global_step,
                         self.word_count,
                         self.grad_norm,
                         self.learning_rate])


    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL

        return sess.run([self.eval_loss,
                         self.predict_count])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        infer_logits, infer_summary, sample_id, sample_words = sess.run([self.infer_logits,
                                                                          self.infer_summary,
                                                                          self.sample_id,
                                                                          self.sample_words])

        return infer_logits, infer_summary, sample_id, sample_words


    def decode(self, sess):
        infer_logits, infer_summary, sample_id, sample_words = self.infer(sess)
        return sample_words, infer_summary



    def initialize_model(self, graph, sess):
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

    def compute_perplexity(self, data, sess, iterator, source_file_placeholder, target_file_placeholder):
        if data == "eval":
            sess.run(iterator.initializer, feed_dict={source_file_placeholder: self.hyper_parameters["eval_source_file"],
                                                      target_file_placeholder: self.hyper_parameters["eval_target_file"]})


            total_loss = 0
            total_predict_count = 0

            while True:
                try:
                    eval_loss, predict_count = self.eval(sess)
                    total_loss += eval_loss * self.hyper_parameters["batch_size"]
                    total_predict_count += predict_count
                except tf.errors.OutOfRangeError:
                    break

            perplexity = np.exp(total_loss/total_predict_count)

            return perplexity