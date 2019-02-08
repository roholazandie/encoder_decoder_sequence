import tensorflow as tf

from nmt.iterator_util import DatasetIterator

UNK_ID = 0


class Model():

    def __init__(self, mode, hyper_parameters):
        self.mode = mode
        self.hyper_parameters = hyper_parameters
        self.iterator = DatasetIterator(hyper_parameters)

    def create_attention_mechanism(self, attention_option, num_units, memory,
                                   source_sequence_length):
        """Create attention mechanism based on the attention_option."""

        # Mechanism
        if attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units,
                                                                    memory,
                                                                    memory_sequence_length=source_sequence_length)
        elif attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units,
                                                                    memory,
                                                                    memory_sequence_length=source_sequence_length,
                                                                    scale=True)
        elif attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                       memory,
                                                                       memory_sequence_length=source_sequence_length)
        elif attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                       memory,
                                                                       memory_sequence_length=source_sequence_length,
                                                                       normalize=True)
        else:
            raise ValueError("Unknown attention option %s" % attention_option)

        return attention_mechanism

    def build_single_cell(self, cell_type, num_units, dropout):
        if cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell(num_units)

        elif cell_type == "lstm":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units)
        else:
            raise ValueError("Unknown cell type: " + str(cell_type))

        if dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=(1.0 - dropout))

        # todo add residual connection
        '''
        single_cell = tf.contrib.rnn.ResidualWrapper(
                single_cell, residual_fn=residual_fn)
        '''

        return cell

    def build_encoder_cell(self, hyper_parameters, num_layers, dropout, num_residual_layers):
        '''
        we can add dropout and residual wrappers
        :param hyper_parameters:
        :param num_layers:
        :param num_residual_layers:
        :return:
        '''
        # dropout (= 1 - keep_prob) is set to 0 during eval and infer
        dropout = dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        num_units = hyper_parameters["num_units"]
        cell_list = []
        for i in range(num_layers):
            # single_cell = tf.contrib.rnn.GRUCell(num_units)
            single_cell = self.build_single_cell(cell_type="gru", num_units=num_units, dropout=dropout)
            cell_list.append(single_cell)

        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def build_decoder_cell(self,
                           encoder_output,
                           encoder_state,
                           num_layers,
                           dropout,
                           num_residual_layers,
                           source_sequence_length):
        if not self.hyper_parameters['has_attention']:
            num_units = self.hyper_parameters["num_units"]
            cell_list = []
            for i in range(num_layers):
                # single_cell = tf.contrib.rnn.GRUCell(num_units)
                single_cell = self.build_single_cell(cell_type="gru",
                                                     num_units=num_units,
                                                     dropout=dropout)
                cell_list.append(single_cell)

            cell = tf.contrib.rnn.MultiRNNCell(cell_list)
            decoder_initial_state = encoder_state

        else:  # with attention
            memory = encoder_output
            num_units = self.hyper_parameters["num_units"]
            cell_list = []
            for i in range(num_layers):
                # single_cell = tf.contrib.rnn.GRUCell(num_units)
                single_cell = self.build_single_cell(cell_type="gru",
                                                     num_units=num_units,
                                                     dropout=dropout)
                cell_list.append(single_cell)

            # output_attention: Python bool. If True (default), the output at each time step is the attention value.
            # This is the behavior of Luong-style attention mechanisms.
            # If False, the output at each time step is the output of cell.
            # This is the behavior of Bhadanau-style attention mechanisms.

            attention_mechanism = self.create_attention_mechanism(attention_option=self.hyper_parameters["attention_option"],
                                                                  num_units=num_units,
                                                                  memory=memory,
                                                                  source_sequence_length=source_sequence_length)

            alignment_history = self.mode == tf.contrib.learn.ModeKeys.INFER  # todo what is this?
            cell = tf.contrib.seq2seq.AttentionWrapper(cell_list,
                                                       attention_mechanism=attention_mechanism,
                                                       attention_layer_size=num_units,
                                                       alignment_history=alignment_history,
                                                       output_attention=True,
                                                       )

            batch_size = self.hyper_parameters["batch_size"]
            if self.hyper_parameters["pass_hidden_state"]:
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

    def get_infer_maximum_iterations(self, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if self.hyper_parameters["target_max_len_infer"]:
            maximum_iterations = self.hyper_parameters["target_max_len_infer"]
        else:
            # TODO(thangluong): add decoding_length_factor flag
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def get_max_time(self, tensor):
        return tensor.shape[1].value or tf.shape(tensor)[1]

    def compute_loss(self, logits, iterator):
        # todo probably i need to add tf.nn.sampled_softmax_loss for performance
        target_output = iterator.target_output_ids
        max_time = self.get_max_time(target_output)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)

        target_weights = tf.sequence_mask(iterator.target_sequence_length, max_time, dtype=tf.float32)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.hyper_parameters["batch_size"])
        return loss


    def build_model(self,
                    iterator,
                    target_vocab_table):
        encoder_type = self.hyper_parameters['encoder_type']

        eos = self.hyper_parameters["eos"]
        sos = self.hyper_parameters["sos"]
        target_vocab_size = self.hyper_parameters["target_vocab_size"]
        target_sos_id = tf.cast(target_vocab_table.lookup(tf.constant(sos)), tf.int32)
        target_eos_id = tf.cast(target_vocab_table.lookup(tf.constant(eos)), tf.int32)

        ### embedding input and output
        # embedding_encoder and embedding_decoder are the whole embedding for input and output languages
        # encoder_embedding_input is the selection of embedding based on input indices with lookup
        embedding_encoder, embedding_decoder = self.iterator.create_embedding_for_encoder_and_decoder(share_vocab=False)
        # the projection layer at the end of encoder and decoder
        # todo is projection weights shared between INFER and TRAIN-EVAL??
        # todo we can replace this with tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size, reuse=reuse)
        with tf.variable_scope("decoder/output_projection"):
            output_layer = tf.layers.Dense(target_vocab_size, use_bias=False, name="output_projection")

        # this is the scope of the encoder_decoder together
        with tf.variable_scope("dynamic_seq2seq"):

            #### build encoder
            encoder_output, encoder_state = self.build_encoder(embedding_encoder, encoder_type, iterator)

            #### build decoder
            logits, decoder_cell_outputs, sample_id, final_context_state = self.build_decoder(embedding_decoder,
                                                                                              encoder_output,
                                                                                              encoder_state,
                                                                                              iterator,
                                                                                              output_layer,
                                                                                              target_eos_id,
                                                                                              target_sos_id)

            # loss
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                self.loss = self.compute_loss(logits, iterator)
            else:
                self.loss = tf.constant(0.0)

        return logits, self.loss, final_context_state, sample_id

    def build_decoder(self, embedding_decoder, encoder_output, encoder_state, iterator, output_layer,
                      target_eos_id, target_sos_id):
        with tf.variable_scope("decoder") as decoder_scope:
            num_decoder_layers = self.hyper_parameters["num_decoder_layers"]
            num_decoder_residual_layers = self.hyper_parameters["num_decoder_residual_layers"]
            dropout = self.hyper_parameters["dropout"]

            cell, decoder_initial_state = self.build_decoder_cell(encoder_output=encoder_output,
                                                                  encoder_state=encoder_state,
                                                                  num_layers=num_decoder_layers,
                                                                  dropout=dropout,
                                                                  num_residual_layers=num_decoder_residual_layers,
                                                                  source_sequence_length=iterator.source_sequence_length)
            # decoder_scope = "same"
            # encoder_scope = "same"

            sample_id = None

            # Train or eval
            decoder_cell_outputs = None
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                decoder_embedding_input = tf.nn.embedding_lookup(embedding_decoder,
                                                                 iterator.target_input_ids)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_input,
                                                           iterator.target_sequence_length)

                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope)

                decoder_cell_outputs = outputs.rnn_output
                logits = output_layer(outputs.rnn_output)


            else:  # Inference mode
                batch_size = self.hyper_parameters["inference_batch_size"]
                start_tokens = tf.fill([batch_size], target_sos_id)
                end_token = target_eos_id

                # we use infer_mode==greedy there are also other options
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, start_tokens, end_token)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state,
                                                          output_layer=output_layer)
                maximum_iterations = self.get_infer_maximum_iterations(iterator.source_sequence_length)
                # Dynamic decoding
                # here we need maximum_iterations because when we do inference we don't know a prior how long
                # the output could be, so we have to have a max iter limit here
                outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=maximum_iterations,
                    swap_memory=True,  # todo ? Whether GPU-CPU memory swap is enabled for this loop.??
                    scope=decoder_scope)

                logits = outputs.rnn_output
                sample_id = outputs.sample_id

        return logits, decoder_cell_outputs, sample_id, final_context_state

    def build_encoder(self, embedding_encoder, encoder_type, iterator):

        with tf.variable_scope("encoder") as encoder_scope:
            encoder_embedding_input = tf.nn.embedding_lookup(embedding_encoder, iterator.source_ids)
            sequence_length = iterator.source_sequence_length
            num_encoder_layers = self.hyper_parameters['num_encoder_layers']
            num_encoder_residual_layers = self.hyper_parameters['num_encoder_residual_layers']
            dropout = self.hyper_parameters["dropout"]
            if encoder_type == 'uni':
                # cell = build_encoder_cell(hyper_parameters, num_decoder_layers, num_residual_layers)
                cell = self.build_encoder_cell(hyper_parameters=self.hyper_parameters,
                                               num_layers=num_encoder_layers,
                                               dropout=dropout,
                                               num_residual_layers=num_encoder_residual_layers
                                               )

                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell,
                                                                  encoder_embedding_input,
                                                                  dtype=tf.float32,
                                                                  sequence_length=sequence_length)

            elif encoder_type == 'bi':
                encoder_output = None

        return encoder_output, encoder_state



