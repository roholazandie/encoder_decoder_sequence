import tensorflow as tf


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm


def train_infer(mode, iterator, hyper_parameters, loss, logits, state=None):

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        word_count = tf.reduce_sum(iterator.source_sequence_length) + tf.reduce_sum(iterator.target_sequence_length)

        predict_count = tf.reduce_sum(iterator.target_sequence_length)

        learning_rate = tf.constant(hyper_parameters["learning_rate"])

        params = tf.trainable_variables()

        # Optimizer
        if hyper_parameters["optimizer"] == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif hyper_parameters["optimizer"] == "adam":
            opt = tf.train.AdamOptimizer(learning_rate)

        # Gradients
        gradients = tf.gradients(loss, params)# colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

        clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients,
                                                                    max_gradient_norm=hyper_parameters["max_gradient_norm"])

        grad_norm_summary = grad_norm_summary
        grad_norm = grad_norm

        global_step = tf.Variable(0, trainable=False)

        train_op = opt.apply_gradients(zip(clipped_grads, params), global_step=global_step)

        #todo add train summary here

        return train_op

    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        eval_loss = loss

    elif mode == tf.contrib.learn.ModeKeys.INFER:
        infer_logits = logits
        final_context_state = state

