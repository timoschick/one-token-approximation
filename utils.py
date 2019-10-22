import io
from typing import Dict, List

import torch
import numpy as np

import log

logger = log.get_logger('root')


def avg(l: List[float]):
    if not l:
        return -1
    return sum(l) / len(l)


def get_word_embedding(word, tokenizer, model, device):
    word_id = tokenizer.convert_tokens_to_ids([word])
    embd = model.embeddings.word_embeddings(torch.tensor(word_id).to(device)).cpu().detach().numpy()[0]
    return embd


def write_embeddings(inferred_embeddings: Dict[str, np.ndarray], output_file: str) -> None:
    with io.open(output_file, 'w', encoding='utf8') as f:
        for word in inferred_embeddings.keys():
            f.write(word + ' ' + ' '.join(str(x) for x in inferred_embeddings[word]) + '\n')


def write_eval(cosine_distances: Dict[int, List[float]], eval_file: str) -> None:
    with open(eval_file, 'w', encoding='utf8') as f:
        f.write('iterations,avg_cosine_distance\n')
        for count in cosine_distances.keys():
            f.write('{},{}\n'.format(count, avg((cosine_distances[count]))))


def token_length(token: str) -> int:
    """
    Returns the actual length of a BPE token without preceding ## characters (BERT) or Ġ characters (RoBERTa)
    :param token: the BPE token
    :return: the number of characters in this token
    """
    return len(token) - (2 if token.startswith('##') else 0) - (1 if token.startswith('Ġ') else 0)


def get_date_string(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
