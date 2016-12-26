# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import json

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def get_qa_set(directory, jsonl_file):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    set_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    set_path = os.path.join(directory, set_name)
    src_path = set_path + '.src'
    targ_path = set_path + '.targ'
    if gfile.Exists(src_path) and gfile.Exists(targ_path):
        return set_path
    with open(jsonl_file, 'r') as qafile, open(src_path,'w') as srcfile, open(targ_path,'w') as targfile:
        for line in qafile:
            lcontent = json.loads(line)
            srcfile.write(lcontent['q'].replace('\n', '') + '\n')
            targfile.write(lcontent['a'].replace('\n', '') + '\n')
    return set_path


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, json_vocab_path):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      json_vocab_path: data file that will be used to create vocabulary.
    """
    if not gfile.Exists(vocabulary_path):
        print("Transform vocabulary to %s" % vocabulary_path)
        with gfile.GFile(json_vocab_path, mode="rb") as f:
            jvocab = json.load(f)
            vocab = jvocab['w2id']
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get)
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    return [vocabulary.get(w, UNK_ID) for w in sentence.strip().split()]


def data_to_token_ids(data_path, target_path, vocabulary_path):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_jsonlbpe_data(data_dir, train_data_file, dev_data_file, vocab_file):
    """Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      train_data_file: jsonl data file.
      dev_data_file: jsonl data file.
      vocab_file: bpe json vocab

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for src training data-set,
        (2) path to the token-ids for target training data-set,
        (3) path to the token-ids for src development data-set,
        (4) path to the token-ids for src development data-set,
        (5) path to the src vocabulary file,
        (6) path to the src vocabulary file.
    """
    if not gfile.Exists(data_dir):
        gfile.MkDir(data_dir)

    # Get wmt data to the specified directory.
    train_path = get_qa_set(data_dir, train_data_file)
    dev_path = get_qa_set(data_dir, dev_data_file)

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab.txt")
    create_vocabulary(vocab_path, vocab_file)

    # Create token ids for the training data.
    src_train_ids_path = train_path + ".src.ids"
    targ_train_ids_path = train_path + ".targ.ids"
    data_to_token_ids(train_path + ".src", src_train_ids_path, vocab_path)
    data_to_token_ids(train_path + ".targ", targ_train_ids_path, vocab_path)

    # Create token ids for the development data.
    src_dev_ids_path = dev_path + ".src.ids"
    targ_dev_ids_path = dev_path + ".targ.ids"
    data_to_token_ids(dev_path + ".src", src_dev_ids_path, vocab_path)
    data_to_token_ids(dev_path + ".targ", targ_dev_ids_path, vocab_path)

    return (src_train_ids_path, targ_train_ids_path,
            src_dev_ids_path, targ_dev_ids_path,
            vocab_path)
