import tensorflow as tf

from nmt.iterator_util import create_embedding_for_encoder_and_decoder

UNK_ID = 0


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def build_encoder_cell(hyper_parameters, num_layers, num_residual_layers):
    '''
    we can add dropout and residual wrappers
    :param hyper_parameters:
    :param num_layers:
    :param num_residual_layers:
    :return:
    '''
    num_units = hyper_parameters["num_units"]
    cell_list = []
    for i in range(num_layers):
        single_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_list.append(single_cell)

    return tf.contrib.rnn.MultiRNNCell(cell_list)


def build_decoder_cell(hyper_parameters, encoder_output, encoder_state, num_layers, num_residual_layers,
                       source_sequence_length):
    if not hyper_parameters['has_attention']:
        num_units = hyper_parameters["num_units"]
        cell_list = []
        for i in range(num_layers):
            single_cell = tf.contrib.rnn.GRUCell(num_units)
            cell_list.append(single_cell)

        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        decoder_initial_state = encoder_state

    else:  # with attention
        memory = encoder_output
        num_units = hyper_parameters["num_units"]
        cell_list = []
        for i in range(num_layers):
            single_cell = tf.contrib.rnn.GRUCell(num_units)
            cell_list.append(single_cell)

        # output_attention: Python bool. If True (default), the output at each time step is the attention value.
        # This is the behavior of Luong-style attention mechanisms.
        # If False, the output at each time step is the output of cell.
        # This is the behavior of Bhadanau-style attention mechanisms.

        attention_mechanism = create_attention_mechanism(attention_option=hyper_parameters["attention_option"],
                                                         num_units=num_units,
                                                         memory=memory,
                                                         source_sequence_length=source_sequence_length)

        cell = tf.contrib.seq2seq.AttentionWrapper(cell_list,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=num_units,
                                                   output_attention=True,
                                                   )

        batch_size = hyper_parameters["batch_size"]
        if hyper_parameters["pass_hidden_state"]:
            decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, tf.float32)

        '''
            if hparams.pass_hidden_state:
          decoder_initial_state = tuple(
          zs.clone(cell_state=es)
          if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
          for zs, es in zip(cell.zero_state(batch_size, dtype), encoder_state))
        else:
          decoder_initial_state = cell.zero_state(batch_size, dtype)
        '''

    return cell, decoder_initial_state


def get_infer_maximum_iterations(hyper_parameters, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hyper_parameters["target_max_len_infer"]:
        maximum_iterations = hyper_parameters["target_max_len_infer"]
    else:
        # TODO(thangluong): add decoding_length_factor flag
        decoding_length_factor = 2.0
        max_encoder_length = tf.reduce_max(source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(
            tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations


def get_max_time(tensor):
    return tensor.shape[1].value or tf.shape(tensor)[1]


def compute_loss(hyper_parameters, logits, iterator):
    target_output = iterator.target_output_ids
    max_time = get_max_time(target_output)

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)

    target_weights = tf.sequence_mask(iterator.target_sequence_length, max_time, dtype=tf.float32)

    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(hyper_parameters["batch_size"])
    return loss

def build_train_model(train_graph,
                      iterator,
                      source_vocab_table,
                      target_vocab_table,
                      hyper_parameters,
                      mode):
    encoder_type = hyper_parameters['encoder_type']
    #mode = "infer"  # or train

    eos = hyper_parameters["eos"]
    sos = hyper_parameters["sos"]
    target_vocab_size = hyper_parameters["target_vocab_size"]
    target_sos_id = tf.cast(target_vocab_table.lookup(tf.constant(sos)), tf.int32)
    target_eos_id = tf.cast(target_vocab_table.lookup(tf.constant(eos)), tf.int32)


    with train_graph.as_default():
        ### embedding input and output
        # embedding_encoder and embedding_decoder are the whole embedding for input and output languages
        # encoder_embedding_input is the selection of embedding based on input indices with lookup
        embedding_encoder, embedding_decoder = create_embedding_for_encoder_and_decoder(share_vocab=False,
                                                 source_vocab_size=hyper_parameters["source_vocab_size"],
                                                 target_vocab_size=hyper_parameters["target_vocab_size"],
                                                 source_embedding_size=hyper_parameters["source_embedding_size"],
                                                 target_embedding_size=hyper_parameters["target_embedding_size"],
                                                 source_vocab_file=hyper_parameters["source_vocab_file"],
                                                 target_vocab_file=hyper_parameters["target_vocab_file"],
                                                 source_embedding_file=hyper_parameters["source_embedding_file"],
                                                 target_embedding_file=hyper_parameters["target_embedding_file"])

        encoder_embedding_input = tf.nn.embedding_lookup(embedding_encoder, iterator.source_ids)
        sequence_length = iterator.source_sequence_length

        target_input = iterator.target_input_ids

        #### build encoder
        num_decoder_layers = hyper_parameters['num_encoder_layers']
        num_residual_layers = hyper_parameters['num_encoder_residual_layers']
        if encoder_type == 'uni':
            cell = build_encoder_cell(hyper_parameters, num_decoder_layers, num_residual_layers)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell,
                                                              encoder_embedding_input,
                                                              dtype=tf.float32,
                                                              sequence_length=sequence_length)

        elif encoder_type == 'bi':
            encoder_output = None

        #### build decoder
        cell, decoder_initial_state = build_decoder_cell(hyper_parameters,
                                                         encoder_output,
                                                         encoder_state,
                                                         hyper_parameters['num_decoder_layers'],
                                                         hyper_parameters['num_encoder_residual_layers'],
                                                         iterator.source_sequence_length)
        decoder_scope = "same"
        encoder_scope = "same"

        sample_id = None
        loss = None

        # Train or eval
        if mode != tf.contrib.learn.ModeKeys.INFER:

            decoder_embedding_input = tf.nn.embedding_lookup(embedding_decoder, target_input)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_input,
                                                       iterator.target_sequence_length)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)

            # Dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=encoder_scope)

            output_layer = tf.layers.Dense(target_vocab_size, use_bias=False, name="output_projection")
            logits = output_layer(outputs.rnn_output)

            loss = compute_loss(hyper_parameters, logits, iterator)



        else:  # Inference mode
            batch_size = hyper_parameters["batch_size"]
            start_tokens = tf.fill([batch_size], target_sos_id)
            end_token = target_eos_id

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, start_tokens, end_token)
            output_layer = tf.layers.Dense(units=target_vocab_size, use_bias=False, name="output_projection")  # resuse
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer=output_layer)

            maximum_iterations = get_infer_maximum_iterations(hyper_parameters, iterator.source_sequence_length)
            # Dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=maximum_iterations,
                swap_memory=True,
                scope=decoder_scope)

            logits = outputs.rnn_output
            sample_id = outputs.sample_id

    return loss, logits, sample_id
