from mynmt.vocabulary import seq2integer, Vocab
from mynmt.iterators import BilingualTextReader
from mxnet.gluon.data import ArrayDataset


def dataset_no_bucketing(source_data_path: str, target_data_path: str,
                         max_seq_len_source: int, max_seq_len_target: int,
                         vocab_source: Vocab, vocab_target: Vocab) -> ArrayDataset:
    source_sentences = []  # List[int]
    target_sentences = []  # List[int]
    for words_source, words_target in BilingualTextReader(path_source=source_data_path,
                                                          path_target=target_data_path,
                                                          max_seq_len_source=max_seq_len_source,
                                                          max_seq_len_target=max_seq_len_target):
        source_sentences.append(seq2integer(words_source, vocab_source, max_seq_len_source))
        target_sentences.append(seq2integer(words_target, vocab_target, max_seq_len_target))
    print(len(target_sentences))
    print(len(source_sentences))
    assert len(source_sentences) == len(target_sentences)
    return ArrayDataset(source_sentences, target_sentences)