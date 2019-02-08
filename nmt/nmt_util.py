import collections


def format_text(words):
  """Convert a sequence words into sentence."""
  if not (hasattr(words, "__len__") or isinstance(words, collections.Iterable)):
    words = [words]
  return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
  """Convert a sequence of bpe words into sentence."""
  words = []
  word = b""
  if isinstance(symbols, str):
    symbols = symbols.encode()
  delimiter_len = len(delimiter)
  for symbol in symbols:
    if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
      word += symbol[:-delimiter_len]
    else:  # end of a word
      word += symbol
      words.append(word)
      word = b""
  return b" ".join(words)


def get_translation(nmt_outputs, sent_id, target_eos, subword_option="bpe"):
    if target_eos:
        target_eos = target_eos.encode("utf-8")

    output = nmt_outputs[sent_id, :].tolist()

    # cut the output to eos char
    if target_eos and target_eos in output:
        output = output[:output.index(target_eos)]

    if subword_option:
        translate = format_bpe_text(output)
    else:
        translate = format_text(output)

    return translate