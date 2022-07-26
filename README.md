# Contrastive Visual Semantic Pretraining Magnifies the Semantics of Natural Language Representations (Replication)

## Introduction
Measure anisotropy for layers of RoBERTa checkpoints. The [original work](https://aclanthology.org/2022.acl-long.217/) has been done on GPT-2 and CLIP.

## Method

I use RoBERTa pre-training checkpoints from [this work](https://aclanthology.org/2021.findings-emnlp.71/). The dataset is the similar to what used by the author. I download the dataset from [here](https://huggingface.co/datasets/stsb_multi_mt) using the Hugging Face API. The calculation of self-similarity (anisotropy) is based on Equation 1 of the original paper.

## Implementation Notes

- I sample strictly one word from each sentence, and I use all sentences. As reported by [the previous work](https://aclanthology.org/2022.acl-long.193/), there is a slight gap between the intra-context and inter-context similarity. Thus, I expect that my sampling method underestimate the anisotropy.
- The total corpus has over 10K sentences. I initially experiment on 1% of the total sentences, and similar trends could already be observed. However, the high anisotropy of the last layer is better observed with more sentences. 
- The pairwise cosine similarity is calculated using tensor operations. For over 10K sampled word embeddings, the matrix could not be hold in the GPU assigned by Colab. When using the CPU for the calculation, this part is slow, but still much faster than without using the tensor operations.
- The sentences have been sorted according to the number of tokens for faster encoding in batches.
- Luckily this dataset is small enough. For larger datasets the API supports read in a stream (e.g. [bookcorpus](https://huggingface.co/datasets/bookcorpus) after this [recent commit](https://github.com/huggingface/datasets/commit/bd8fd273000a02bae960a32a92d543ba3eab1bed)), but I could not make it work. After being downloaded, the dataset is cached in the space temporarily assigned by Colab.
- Use the `AutoModel` class instead of the `AutoModelForMaskedLM` class to get the representations. Enable `output_hidden_states` to get the hidden layer from all 13 layers.
- When intializing an empty list of lists, do not use `[[]] * n`. This initialization makes each sublist shared, so that appending to any one of the sublists would also append to all others. 
- I reuse model checkpoints from the [yearly-bias repository](https://github.com/kt2k01/yearly-bias) saved in my Google Drive, hence the directory name in the code.

## Results and Discussion

The result is saved in `anisotropy.pkl`, and the plot could be found in `anisotropy.ipynb`. The resulting anisotropy lies between the GPT-2 and CLIP. The result makes sense, as the original paper mentions that BERT produces clusters in the space, so isotropy to some extent (lower anisotropy) could be expected. However, in the last two layers of RoBERTa, there is still a jump in anisotropy. The anisotropy does not monotonically increase with the pre-training process. For each checkpoint, the words for calculation are resampled. Overall, the anisotropy increases for the last 2 layers, but decreases for other layers during pre-training.