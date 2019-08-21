import argparse
from pathlib import Path

from mynmt.minibatch import dataset_no_bucketing
from mynmt.vocabulary import build_and_save_vocabularies


def add_parser_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(func=train)

    data_params = parser.add_argument_group('Data preparation parameters')
    data_params.add_argument('--train-source', type=str, required=True)
    data_params.add_argument('--train-target', type=str, required=True)
    data_params.add_argument('--max-seq-len-source', type=int, default=100)
    data_params.add_argument('--max-seq-len-target', type=int, default=100)

    output_params = parser.add_argument_group("Output parameters")
    output_params.add_argument('--output-dir', type=str, required=True)

    return parser


def train(args: argparse.Namespace):
    Path(args.output_dir).mkdir(parents=True)

    vocab_source, vocab_target = build_and_save_vocabularies(train_source=args.train_source,
                                                             train_target=args.train_target,
                                                             output_dir=args.output_dir,
                                                             max_seq_len_source=args.max_seq_len_source,
                                                             max_seq_len_target=args.max_seq_len_target)
    data = dataset_no_bucketing(source_data_path=args.train_source,
                                target_data_path=args.train_target,
                                max_seq_len_source=args.max_seq_len_source,
                                max_seq_len_target=args.max_seq_len_target,
                                vocab_source=vocab_source,
                                vocab_target=vocab_target)
    q, w = data
    print(len(q))
    print(len(w))

    
