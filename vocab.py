import argparse
from collections import Counter

from transformers import BertTokenizer, GPT2Tokenizer
from typing import Dict, List, Set

import log
from ota import MODELS

logger = log.get_logger('root')


def load_vocab(path: str) -> List[str]:
    with open(path, 'r', encoding='utf8') as f:
        vocab = f.read().splitlines()
    return vocab


def load_vocab_with_counts(path: str, min_freq=-1, max_freq=-1) -> Dict[str, int]:
    logger.info('Loading vocab from {} with minimum count {}'.format(path, min_freq))
    vocab = Counter()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, count = line.split()
            if (min_freq <= 0 or int(count) >= min_freq) and (max_freq <= 0 or int(count) <= max_freq):
                vocab[word] = int(count)

    logger.info('The loaded vocab contains {} words'.format(len(vocab)))
    return vocab


def get_difference(vocab_path: str, model: str, model_cls: str) -> List[str]:
    """
    Returns the difference between the words in a vocabulary file and the words in the vocabulary of a given
    Transformer model.
    :param vocab_path: the path to the vocabulary file
    :param model: the path to the model
    :param model_cls: the model class (currently supported: "bert" or "roberta")
    :return: the difference as a list of words
    """
    vocab_with_counts = load_vocab_with_counts(vocab_path)
    vocab = set(vocab_with_counts.keys())

    model_cls, tokenizer_cls = MODELS[model_cls]
    tokenizer = tokenizer_cls.from_pretrained(model)

    if isinstance(tokenizer, BertTokenizer):
        model_vocab = tokenizer.vocab.keys()
    elif isinstance(tokenizer, GPT2Tokenizer):
        model_vocab = set(tokenizer.encoder.keys())
        model_vocab.update(w[1:] for w in model_vocab if w.startswith('Ġ'))
    else:
        raise ValueError('Access to vocab is currently only implemented for BertTokenizer and GPT2Tokenizer')

    logger.info('Vocab sizes: file = {}, model = {}'.format(len(vocab), len(model_vocab)))
    vocab -= model_vocab
    logger.info('Size of vocab difference = {}'.format(len(vocab)))

    vocab = list(vocab)
    vocab.sort(key=lambda x: vocab_with_counts[x], reverse=True)
    return vocab


def split_vocab(path: str, parts: int):
    vocab = load_vocab(path)
    part_size = int(len(vocab) / parts) + 1

    logger.info("Splitting vocab with {} words into {} parts with size {}".format(len(vocab), parts, part_size))
    vocab_splitted = [vocab[part_size * i: part_size * (i + 1)] for i in range(parts)]
    assert sum(len(x) for x in vocab_splitted) == len(vocab)

    for idx, vocab_part in enumerate(vocab_splitted):
        logger.info("Vocab part {} has size {}".format(idx, len(vocab_part)))
        write_vocab(vocab_part, path + str(idx))


def write_vocab(lines: List[str], file: str) -> None:
    with open(file, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--output', default=None, type=str, required=True)
    parser.add_argument('--model', default='bert-base-uncased', type=str)
    parser.add_argument('--model_cls', default='bert', type=str, choices=['bert', 'roberta'])
    parser.add_argument('--parts', default=0, type=int)
    args = parser.parse_args()

    vocab = get_difference(args.input, args.model, args.model_cls)
    write_vocab(vocab, args.output)

    if args.parts > 0:
        split_vocab(args.output, args.parts)


if __name__ == "__main__":
    main()
