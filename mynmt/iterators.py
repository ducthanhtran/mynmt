from typing import List, Tuple

from mynmt.utils import smart_open


class TextReader:
    def __init__(self, path: str, max_seq_len: int):
        self.path = path
        self.max_seq_len = max_seq_len

    def __iter__(self) -> List[str]:
        with smart_open(self.path) as text_data:
            for line in text_data:
                words = line.split()
                if len(words) > self.max_seq_len:
                    continue
                yield words


class BilingualTextReader:
    def __init__(self, path_source: str, path_target: str, max_seq_len_source: int, max_seq_len_target: int):
        self.path_source = path_source
        self.path_target = path_target
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target

    def __iter__(self) -> Tuple[List[str], List[str]]:
        with smart_open(self.path_source) as source, smart_open(self.path_target) as target:
            for line_source, line_target in zip(source, target):
                words_source = line_source.split()
                words_target = line_target.split()
                if len(words_source) > self.max_seq_len_source or len(words_target) > self.max_seq_len_target:
                    continue
                yield words_source, words_target
