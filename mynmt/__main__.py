import argparse
import mynmt.train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train')
    mynmt.train.add_parser_arguments(train_parser)

    args = parser.parse_args()
    args.func(args)