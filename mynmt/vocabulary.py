import gzip
from json import dump
from itertools import chain
from typing import Dict, List, Tuple

import mynmt.constants as consts
from mynmt.constants import PAD
from mynmt.iterators import BilingualTextReader


Vocab = Dict[str, int]

def build_and_save_vocabularies(train_source: str, train_target: str, output_dir: str,
                                max_seq_len_source: int, max_seq_len_target: int) -> Tuple[Vocab, Vocab]:
    vocab_source, vocab_target = _build_vocabularies(path_source=train_source, path_target=train_target,
                                                     special_tokens_vocab_source=consts.SPECIAL_VOCABULARY_SOURCE,
                                                     special_tokens_vocab_target=consts.SPECIAL_VOCABULARY_TARGET,
                                                     max_seq_len_source=max_seq_len_source,
                                                     max_seq_len_target=max_seq_len_target)
    _save_vocabularies(vocab_source, output_dir, consts.SOURCE_VOCAB_FILE_NAME)
    _save_vocabularies(vocab_target, output_dir, consts.TARGET_VOCAB_FILE_NAME)
    return vocab_source, vocab_target


def _build_vocabularies(path_source: str, path_target: str,
                        special_tokens_vocab_source: Vocab, special_tokens_vocab_target: Vocab,
                        max_seq_len_source: int, max_seq_len_target: int) -> Tuple[Vocab, Vocab]:
    unique_source_words = set()
    unique_target_words = set()
    for words_source, words_target in BilingualTextReader(path_source=path_source, path_target=path_target,
                                                          max_seq_len_source=max_seq_len_source,
                                                          max_seq_len_target=max_seq_len_target):
        unique_source_words.update(words_source)
        unique_target_words.update(words_target)

    vocab_source = {word:index for index, word in enumerate(chain(special_tokens_vocab_source, unique_source_words))}
    vocab_target = {word: index for index, word in enumerate(chain(special_tokens_vocab_target, unique_target_words))}

    return vocab_source, vocab_target


def _save_vocabularies(vocabulary: Vocab, output_dir: str, output_fname: str) -> None:
    vocab_file = "{}/{}".format(output_dir, output_fname)
    with gzip.open(vocab_file, mode="wt", encoding='utf-8') as file_out:
        dump(vocabulary, file_out)


def seq2integer(sequence: List[str], vocabulary: Vocab, max_seq_len: int) -> List[int]:
    mapped = list(map(vocabulary.get, sequence))
    return mapped + [vocabulary[PAD]] * (max_seq_len - len(mapped))
