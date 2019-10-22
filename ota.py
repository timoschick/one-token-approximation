import random
import io
import argparse
import time
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import scipy.spatial.distance
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, GPT2Tokenizer

import utils
import log
from modeling import InputPreparator, OverwriteableEmbedding

logger = log.get_logger('root')

MODELS = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer)
}


def verify_args(args):
    if args.pmin > args.pmax:
        raise ValueError('pmin must be less than pmax, got pmin={}, pmax={}'.format(args.pmin, args.pmax))
    if args.smin > args.smax:
        raise ValueError('smin must be less than smax, got smin={}, smax={}'.format(args.smin, args.smax))
    if (not args.word and not args.words) or (args.word and args.words):
        raise ValueError('Either a single word or a file containing words must be given via --word or --words')
    if args.prefix != '' and not args.prefix.endswith(' '):
        raise ValueError('The prefix must either be empty or end with a space, got "{}"'.format(args.prefix))
    if args.suffix != '' and not args.suffix.startswith(' '):
        raise ValueError('The suffix must either be empty or start with a space, got "{}"'.format(args.suffix))


def load_words(word, word_file):
    if word:
        return [word]
    else:
        with io.open(word_file, 'r', encoding='utf8') as f:
            return f.read().splitlines()


def initialize_embeddings(tokens: List[List[str]], embeddings: List[List[np.ndarray]], strategy: str) -> torch.Tensor:
    # embeddings and tokens are lists of shape batch_size x nr_of_tokens
    batch_size = len(embeddings)
    embedding_dim = embeddings[0][0].shape[0]

    logger.info('Initializing embeddings of shape {} x {} (strategy={})'.format(batch_size, embedding_dim, strategy))

    embeddings_sum = torch.zeros(batch_size, embedding_dim)
    if len(embeddings[0]) == 1:
        return embeddings_sum

    if strategy == 'sum':
        for idx, token_embeddings in enumerate(embeddings):
            for embedding in token_embeddings:
                embeddings_sum[idx] += torch.tensor(embedding)
            embeddings_sum[idx] /= len(token_embeddings)

    elif strategy == 'wsum':
        for idx, token_embeddings in enumerate(embeddings):
            word_tokens = tokens[idx]
            for token_idx, token in enumerate(word_tokens):
                embeddings_sum[idx] += torch.tensor(token_embeddings[token_idx]) * utils.token_length(token)
            embeddings_sum[idx] /= sum(utils.token_length(token) for token in word_tokens)

    return embeddings_sum


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_file', default=None, type=str, required=True)

    parser.add_argument('--word', type=str, required=False)
    parser.add_argument('--words', type=str, required=False)

    parser.add_argument('--model', default='bert-base-uncased', type=str)
    parser.add_argument('--model_cls', default='bert', type=str, choices=['bert', 'roberta'])

    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--prefix', '-p', default='', type=str)
    parser.add_argument('--suffix', '-s', default=' .', type=str)

    parser.add_argument('--smin', default=1, type=int,
                        help='Minimum number of random tokens to append to the training string as suffix')

    parser.add_argument('--smax', default=1, type=int,
                        help='Maximum number of random tokens to append to the training string as suffix')

    parser.add_argument('--pmin', default=1, type=int,
                        help='Minimum number of random tokens to prepend to the training string as prefix')

    parser.add_argument('--pmax', default=1, type=int,
                        help='Maximum number of random tokens to prepend to the training string as prefix')

    parser.add_argument('--objective', '-o', default='both',
                        choices=['first', 'left', 'right', 'both'],
                        help='Training objective: Whether to minimize only the distance of '
                             'embeddings for the first token (i.e. [CLS]), for all words to '
                             'the left, to the right or both to the left and right.')

    parser.add_argument('--eval_file', default=None, type=str)

    parser.add_argument('--eval_steps', default=[1, 10] + [100 * i for i in range(1, 51)], type=int, nargs='+',
                        help='The numbers of training steps after which the average cosine distance over the'
                             'entire list of words is computed and stored in the eval file')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float)

    parser.add_argument('--init', default='sum', choices=['sum', 'wsum', 'zeros'],
                        help='Initialization strategy for the embedding to be inferred. '
                             'With "zeros", the embedding is initialized as a zero vector. '
                             'With "sum", it is initialized as the sum of all embeddings '
                             'of the BPE of the original word.'
                             'With "wsum", it is initialized as the weighted sum of all'
                             'token embeddings, where the weight is based on each token\'s length')

    args = parser.parse_args()
    verify_args(args)

    words = load_words(args.word, args.words)
    uses_randomization = args.smax > 0 or args.pmax > 0

    logger.info('Inferring embeddings for {} words, first 10 are: {}'.format(len(words), words[:10]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_cls, tokenizer_cls = MODELS[args.model_cls]

    tokenizer = tokenizer_cls.from_pretrained(args.model)
    input_preparator = InputPreparator(tokenizer, **vars(args))

    model = model_cls.from_pretrained(args.model, output_hidden_states=True)
    model.to(device)
    model.eval()  # we don't want any dropout so we use eval mode
    for param in model.parameters():
        param.requires_grad = False

    def tokenize_with_optional_space(word):
        if isinstance(tokenizer, GPT2Tokenizer):
            return tokenizer.tokenize(word, add_prefix_space=True)
        return tokenizer.tokenize(word)

    # group words based on their number of tokens
    words_by_token_size = defaultdict(list)
    for word in words:
        num_tokens = len(tokenize_with_optional_space(word))
        if num_tokens > 0:
            words_by_token_size[num_tokens].append(word)

    token_sizes = list(words_by_token_size.keys())
    logger.info('Found words with the following token sizes: {}'.format(token_sizes))

    dists = []
    token_sizes_idx = -1
    words_for_token_size = []

    batch_idx = 0

    inferred_embeddings = {}
    cosine_distances = defaultdict(list)

    word_embeddings = model.embeddings.word_embeddings
    model.embeddings.word_embeddings = OverwriteableEmbedding(word_embeddings, overwrite_fct=None)

    while True:
        batch = words_for_token_size[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]

        if not batch:
            if token_sizes_idx < len(token_sizes) - 1:
                token_sizes_idx += 1
                token_size = token_sizes[token_sizes_idx]
                words_for_token_size = words_by_token_size[token_size]
                batch_idx = 0
                logger.info('Processing all words that consist of {} tokens (found {} words, first 10 are: {})'.format(
                    token_size, len(words_for_token_size), words_for_token_size[:10]))
                continue
            else:
                break

        logger.info(
            'Processing words {} - {} of {}'.format(batch_idx * args.batch_size + 1,
                                                    batch_idx * args.batch_size + len(batch),
                                                    len(words_for_token_size)))

        tokens = [tokenize_with_optional_space(wrd) for wrd in batch]
        embeddings = [[utils.get_word_embedding(wordpart, tokenizer, model, device) for wordpart in
                       tokenize_with_optional_space(wrd)] for wrd in batch]

        print(len(embeddings[0]))
        print([len(x) for x in embeddings])

        init_vals = initialize_embeddings(tokens, embeddings, args.init)
        optim_vars = torch.tensor(init_vals, requires_grad=True)
        optimizer = torch.optim.Adam([optim_vars], lr=args.learning_rate)

        input_gold, input_inference, index_to_optimize, layers_gold = None, None, None, None

        if not uses_randomization:
            input_gold, input_inference, index_to_optimize = input_preparator.prepare_batch(batch)

            print('IGS:', input_gold.tokens.shape, 'IIS', input_inference.tokens.shape)

            _, _, layers_gold = model(input_gold.tokens.to(device), input_gold.segments.to(device))
            layers_gold = [layer.detach() for layer in layers_gold]

        def overwrite_fct(embds):
            for i in range(embds.shape[0]):
                embds[i, index_to_optimize, :] = optim_vars[i]
            return embds

        start = time.time()
        logger.info(' ' * 79 + ' '.join('{:6s}'.format(word[:6]) for word in batch[:10]))

        for iteration in range(1, args.iterations + 1):

            if uses_randomization:
                input_gold, input_inference, index_to_optimize = input_preparator.prepare_batch(batch)
                with torch.no_grad():
                    _, _, layers_gold = model(input_gold.tokens.to(device), input_gold.segments.to(device))
                layers_gold = [layer.detach() for layer in layers_gold]

            model.embeddings.word_embeddings.overwrite_fct = overwrite_fct
            _, _, layers = model(input_inference.tokens.to(device), input_inference.segments.to(device))
            model.embeddings.word_embeddings.overwrite_fct = None

            loss = nn.MSELoss()
            loss_val = torch.tensor(0, dtype=torch.float).to(device)

            for idx in range(model.config.num_hidden_layers):

                if args.objective == 'first':
                    loss_val += loss(layers[idx][:, 0, :], layers_gold[idx][:, 0, :])

                if args.objective == 'left' or args.objective == 'both':
                    loss_val += loss(layers[idx][:, :index_to_optimize, :],
                                     layers_gold[idx][:, :index_to_optimize, :])

                if args.objective == 'right' or args.objective == 'both':
                    loss_val += loss(layers[idx][:, index_to_optimize + 1:, :],
                                     layers_gold[idx][:, index_to_optimize + len(embeddings[0]):, :])

            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

            now = time.time()
            elapsed_time = (now - start)

            do_eval = args.eval_file is not None and iteration in args.eval_steps

            if (iteration == 1 or iteration % 100 == 0) or do_eval:
                batch_dists = []
                if len(embeddings[0]) == 1:
                    for idx, token_embeddings in enumerate(embeddings):
                        inferred_embedding = optim_vars[idx].cpu().detach().numpy()
                        if np.linalg.norm(inferred_embedding) > 0:
                            cosine_distance = scipy.spatial.distance.cosine(token_embeddings[0],
                                                                            inferred_embedding)
                            batch_dists.append(cosine_distance)

                if do_eval:
                    cosine_distances[iteration] += batch_dists

                cosine_string = ' '.join(['{:6.2f}'.format(dist) for dist in batch_dists][:10])
                steps_per_second = iteration / elapsed_time
                remaining_words = len(words) - len(inferred_embeddings) - len(batch)
                remaining_time_for_other_batches = (
                                                           remaining_words / args.batch_size) * args.iterations / steps_per_second
                remaining_time_for_this_batch = (args.iterations - iteration) / steps_per_second
                remaining_time = int(remaining_time_for_this_batch + remaining_time_for_other_batches)

                logger.info('step: {:4d} loss: {:8.6f} cosine: {:6.2f} steps/s: {:6.2f} ETR: {:8s} sample: {}'.format(
                    iteration, loss_val.item(), utils.avg(batch_dists), steps_per_second,
                    utils.get_date_string(remaining_time), cosine_string))

        for idx, token_embeddings in enumerate(embeddings):

            word = batch[idx]

            inferred_embedding = optim_vars[idx].cpu().detach().numpy()
            if len(embeddings[0]) == 1:
                final_cosine_distance = scipy.spatial.distance.cosine(token_embeddings[0], inferred_embedding)
                dists.append(final_cosine_distance)
            inferred_embeddings[word] = inferred_embedding

        batch_idx += 1

    logger.info('Overall average cosine distance: {}'.format(utils.avg(dists)))

    utils.write_embeddings(inferred_embeddings, args.output_file)
    if args.eval_file is not None:
        utils.write_eval(cosine_distances, args.eval_file)


if __name__ == "__main__":
    main()
