import tensorflow as tf
import os
import codecs

def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size

def check_vocab(vocab_file, out_dir, check_special_token, sos, eos, unk):
    if tf.gfile.Exists(vocab_file):
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            # Verify if the vocab starts with unk, sos, eos
            # If not, prepend those tokens & generate a new vocab file
            # if not unk: unk = UNK
            # if not sos: sos = SOS
            # if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file '%s' does not exist." % vocab_file)

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def get_hyper_parameters():
    hyper_parameters = {}
    hyper_parameters["batch_size"] = 10
    hyper_parameters["inference_batch_size"] = 2
    hyper_parameters["sos"] = "<s>"
    hyper_parameters["eos"] = "</s>"
    hyper_parameters["unk"] = "<unk>"
    hyper_parameters["source_max_len_train"] = 20
    hyper_parameters["target_max_len_train"] = 10
    hyper_parameters["source_max_len_infer"] = 20
    hyper_parameters["target_max_len_infer"] = 10

    hyper_parameters["source_vocab_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/vocab.bpe.32000.en"
    hyper_parameters["target_vocab_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/vocab.bpe.32000.de"
    #hyper_parameters["source_vocab_file"] = "'/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/vocab.bpe.32000.de'"
    #hyper_parameters["target_vocab_file"] = 36549
    hyper_parameters["train_source_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/train.tok.clean.bpe.32000.en"
    hyper_parameters["train_target_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/train.tok.clean.bpe.32000.de"

    #todo add pretrained embedding files
    hyper_parameters["source_embedding_file"] = None #here we dont provide pretrained embedding so it will train it end-to-end
    hyper_parameters["target_embedding_file"] = None
    hyper_parameters["source_embedding_size"] = 300
    hyper_parameters["target_embedding_size"] = 300

    hyper_parameters["eval_source_file"] = ""
    hyper_parameters["eval_target_file"] = ""
    hyper_parameters["test_source_file"] = ""
    hyper_parameters["test_target_file"] = ""

    hyper_parameters['encoder_type'] = "uni"
    hyper_parameters["residual"] = True
    hyper_parameters['num_encoder_layers'] = 3
    #todo residual layers for GNMT num_encoder_residual_layers = hparams.num_encoder_layers - 2
    hyper_parameters["share_vocab"] = False
    hyper_parameters['num_encoder_residual_layers'] = hyper_parameters['num_encoder_layers'] - 1
    hyper_parameters["num_decoder_layers"] = 3
    hyper_parameters["num_decoder_residual_layers"] = hyper_parameters["num_decoder_layers"] - 1
    hyper_parameters["dropout"] = 0.2
    hyper_parameters["num_units"] = 50
    hyper_parameters['has_attention'] = False
    hyper_parameters["attention_option"] = "luong"
    hyper_parameters["pass_hidden_state"] = True
    hyper_parameters["target_vocab_size"] = 0
    hyper_parameters["target_max_len_infer"] = 20

    hyper_parameters["max_gradient_norm"] = None
    hyper_parameters["optimizer"] = "adam"
    hyper_parameters["learning_rate"] = 0.1# should be 1
    hyper_parameters["max_gradient_norm"] = 5.0
    hyper_parameters["num_training_steps"] = 1000#should be 340000

    ####add logic to load more hyperparameters

    hyper_parameters["source_vocab_size"], hyper_parameters["source_vocab_file"] = check_vocab(hyper_parameters["source_vocab_file"],
                out_dir="tmp/",
                check_special_token=True,
                sos=hyper_parameters["sos"],
                eos=hyper_parameters["eos"],
                unk=hyper_parameters["unk"])

    if hyper_parameters["share_vocab"]:
        hyper_parameters["target_vocab_size"], hyper_parameters["target_vocab_file"] = hyper_parameters[
                                                                                           "source_vocab_size"], \
                                                                                       hyper_parameters[
                                                                                           "source_vocab_file"]
    else:
        hyper_parameters["target_vocab_size"], hyper_parameters["target_vocab_file"] = check_vocab(hyper_parameters["target_vocab_file"],
                                                               out_dir="tmp/",
                                                               check_special_token=True,
                                                               sos=hyper_parameters["sos"],
                                                               eos=hyper_parameters["eos"],
                                                               unk=hyper_parameters["unk"])

    # todo partitioning the embedding instead of using embedding files





    return hyper_parameters