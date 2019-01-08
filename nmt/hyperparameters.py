

def get_hyperparameters():
    hyper_parameters = {}
    hyper_parameters["batch_size"] = 1
    hyper_parameters["sos"] = "<s>"
    hyper_parameters["eos"] = "</s>"
    hyper_parameters["source_max_len"] = 20
    hyper_parameters["target_max_len"] = 10

    hyper_parameters["source_vocab_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/vocab.bpe.32000.en"
    hyper_parameters["target_vocab_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/vocab.bpe.32000.de"
    hyper_parameters["source_vocab_size"] = 36549
    hyper_parameters["target_vocab_size"] = 36549
    hyper_parameters["train_source_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/train.tok.clean.bpe.32000.en"
    hyper_parameters["train_target_file"] = "/home/rohola/Codes/Python/nmt/nmt/scripts/wmt16_de_en/train.tok.clean.bpe.32000.de"
    hyper_parameters["source_embedding_file"] = None
    hyper_parameters["target_embedding_file"] = None
    hyper_parameters["source_embedding_size"] = 300
    hyper_parameters["target_embedding_size"] = 300

    hyper_parameters['encoder_type'] = "uni"
    hyper_parameters['num_encoder_layers'] = 3
    hyper_parameters['num_encoder_residual_layers'] = 1
    hyper_parameters["num_units"] = 50
    hyper_parameters['num_decoder_layers'] = 3
    hyper_parameters['has_attention'] = False
    hyper_parameters["target_vocab_size"] = 36549
    hyper_parameters["target_max_len_infer"] = 20

    hyper_parameters["max_gradient_norm"] = None
    hyper_parameters["optimizer"] = "adam"
    hyper_parameters["learning_rate"] = 0.01
    hyper_parameters["max_gradient_norm"] = 5.0

    return hyper_parameters