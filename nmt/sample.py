# import tensorflow as tf
#
# UNK_ID = 0
# source_vocab_file = "vocab.txt"
# features = tf.constant(["emerson", "lake", "and", "palmer"])
#
# source_vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=source_vocab_file,
#                                                                      default_value=UNK_ID)
#
# ids = source_vocab_table.lookup(features)
#
# with tf.Session() as sess:
#     tf.tables_initializer().run()
#
#     print(ids.eval())


def func(x, *args, **kwargs):
    print(x)
    print(args)
    print(kwargs["l"])


if __name__ == "__main__":
    func(6, 7, 9, 8, 9, k=0, l=8)