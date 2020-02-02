# One-Token Approximation

While traditional word embedding algorithms (e.g., Word2Vec, Glove) assign a **single** embedding to each word, pretrained language models (e.g., BERT, RoBERTa, XLNet, T5) typically represent words as sequences of subword tokens. For example, BERT represents the word `strawberries` as two tokens `straw` and `##berries`. Transferring ideas and algorithms from traditional embeddings to contextualized embeddings may therefore raise questions like the following:

> How would the embedding of "strawberries" (or any other multi-token word) look like in BERT's embedding space if the word was represented by a single token?

*One-Token Approximation* (OTA) can be used to answer this question. More details can be found [here](https://arxiv.org/abs/1904.06707).

## Dependencies

All dependencies can be found in `environment.yml`. If you use conda, simply type
```
conda env create -f environment.yml
```
to create a new environment with all required packages installed.

## Usage

To obtain One-Token Approximations for multi-token words, run the following command:

```
python3 ota.py --words WORDS --output_file OUTPUT_FILE --model_cls MODEL_CLS --model MODEL --iterations ITERATIONS
```
where
- `WORDS` is the path to a file containing all words for which one-token approximations should be computed (with each line containing exactly one word);
- `OUTPUT_FILE` is the path to a file where all one-token approximations are saved (in the format `<WORD> <EMBEDDING>`);
- `MODEL_CLS` is either `bert` or `roberta` (the script currently does not support other pretrained language models);
- `MODEL` is either the name of a pretrained model from the [Hugging Face Transformers Library](https://github.com/huggingface/transformers) (e.g., `bert-base-uncased`) or the path to a finetuned model;
- `ITERATIONS` is the number of iterations for which to perform OTA. For BERT, 4000 iterations generally give good results; for RoBERTa, we found that much better results can be obtained by increasing the number of iterations to 8000.

For additional parameters, check the source code of `ota.py` or run `python3 ota.py --help`. 

## Citation

If you make use of One-Token Approximation, please cite the following paper:

```
@inproceedings{schick2020rare,
  title={Rare words: A major problem for contextualized representation and how to fix it by attentive mimicking},
  author={Schick, Timo and Sch{\"u}tze, Hinrich},
  url="https://arxiv.org/abs/1904.06707",
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
