import codecs
import tensorflow as tf

def load_data(inference_input_file, hyper_parameters=None):
    with codecs.getdecoder("utf-8")(tf.gfile.GFile(inference_input_file, mode="rb")) as gfile_reader:
        inference_data = gfile_reader.read().splitlines()


    if hyper_parameters and hyper_parameters["inference_indices"]:
        inference_data = [inference_data[i] for i in hyper_parameters["inference_indices"]]


    return inference_data