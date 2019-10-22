import random
import torch
from typing import Callable, List, Tuple

from torch import Tensor
from torch.nn import Module, Embedding

from transformers import PreTrainedTokenizer, BertTokenizer, GPT2Tokenizer

import log

logger = log.get_logger('root')


def default_filter(x: str) -> bool:
    return not x.startswith('[') and not x.startswith('#')


class OverwriteableEmbedding(Module):

    def __init__(self, embedding: Embedding, overwrite_fct):
        super().__init__()
        self.embedding = embedding
        self.overwrite_fct = overwrite_fct

    def forward(self, input: Tensor):
        embds = self.embedding(input)
        if self.overwrite_fct is not None:
            embds = self.overwrite_fct(embds)
        return embds


class OTAInput:
    def __init__(self, tokens, segments=None, mask=None):
        self.tokens = tokens
        self.segments = segments if segments is not None else torch.zeros(self.tokens.shape, dtype=torch.long)
        self.mask = mask if mask is not None else torch.ones(self.tokens.shape, dtype=torch.long)

    def get_length(self) -> int:
        return self.tokens.shape[0]

    @staticmethod
    def stack(inputs: List['OTAInput']) -> 'OTAInput':
        max_seq_length = max(x.get_length() for x in inputs)
        for inp in inputs:
            # zero-pad up to the sequence length
            padding = torch.tensor([0] * (max_seq_length - inp.get_length()), dtype=torch.long)
            inp.tokens = torch.cat((inp.tokens, padding), dim=0)
            inp.segments = torch.cat((inp.segments, padding), dim=0)
            inp.mask = torch.cat((inp.mask, padding), dim=0)

        stacked_tokens = torch.stack([x.tokens for x in inputs])
        stacked_segments = torch.stack([x.segments for x in inputs])
        stacked_masks = torch.stack([x.mask for x in inputs])
        return OTAInput(stacked_tokens, stacked_segments, stacked_masks)


class InputPreparator:
    def __init__(self, tokenizer: PreTrainedTokenizer, filter_callable: Callable[[str], bool] = default_filter,
                 prefix: str = '', suffix: str = ' .', pmin: int = 0, pmax: int = 0, smin: int = 0, smax: int = 0,
                 seed: int = 1234, eval_sentence: str = None, **_):
        self.tokenizer = tokenizer

        if isinstance(tokenizer, BertTokenizer):
            vocab = tokenizer.vocab.keys()
        elif isinstance(tokenizer, GPT2Tokenizer):
            vocab = tokenizer.encoder.keys()
        else:
            raise ValueError('Access to vocab is currently only implemented for BertTokenizer and GPT2Tokenizer')

        self.words = [x for x in vocab if not filter_callable or filter_callable(x)]
        self.prefix = tokenizer.tokenize(prefix)
        self.suffix = tokenizer.tokenize(suffix)
        self.pmin = pmin
        self.pmax = pmax
        self.smin = smin
        self.smax = smax
        self.eval_sentence = eval_sentence

        if seed:
            random.seed(seed)

    def generate_random_word(self) -> str:
        return random.choice(self.words)

    def prepare_batch(self, batch: List[str]) -> Tuple[OTAInput, OTAInput, int]:

        prefix, suffix = self._create_infixes()
        index_to_optimize = len(prefix) + 1

        inputs_gold = []
        inputs_inference = []

        for word in batch:
            if isinstance(self.tokenizer, GPT2Tokenizer):
                word_toks = self.tokenizer.tokenize(word, add_prefix_space=True)
            else:
                word_toks = self.tokenizer.tokenize(word)
            inputs_gold.append(self._prepare_input(prefix, word_toks, suffix))
            inputs_inference.append(self._prepare_input(prefix, [self.tokenizer.mask_token], suffix))

        return OTAInput.stack(inputs_gold), OTAInput.stack(inputs_inference), index_to_optimize

    def _create_infixes(self):
        num_prefix_words = random.randint(self.pmin, self.pmax)
        num_suffix_words = random.randint(self.smin, self.smax)

        prefix = list(self.prefix) + [self.generate_random_word() for _ in range(num_prefix_words)]
        suffix = [self.generate_random_word() for _ in range(num_suffix_words)] + list(self.suffix)

        logger.debug('Randomly sampled template: {} <WORD> {}'.format(prefix, suffix))
        return prefix, suffix

    def _prepare_input(self, prefix, word, suffix) -> OTAInput:
        token_ids = self.tokenizer.encode(prefix + word + suffix, add_special_tokens=True)
        tokens_tensor = torch.tensor(token_ids)
        return OTAInput(tokens=tokens_tensor)
