# Bridging Background Knowledge Gaps in Translation with Automatic Explicitation
This repository contains our **WikiExpl** dataset, a semi-automatic collection of naturally occurring explicitations in Wikipedia bitext corpus annotated by human translators, from our EMNLP 2023 main conference paper ([arXiv](https://arxiv.org/abs/2312.01308)).

The `json` files contain the candidates extracted by our detection algorithm.
Each candidate is annotated by three annotators and we assign the label based on the majority vote. 
We consider the candidates as final explicitation if two or more annotators agree.
The list of final explicitation is in `expl_idx_list`. We merge the annotated span of explicitation from different annotators by maximizing the span coverage.

We provide simple tools for easy exploration:
```bash
$ python show.py
```
The output example :
![output_example](doc/ex1.png)
Here the red part in the source text (green) is that which is to be performed explicitation in the corresponding target translation, and the red part in the target text (blue) is its explicitation.

## Reference
```
@inproceedings{han-etal-2023-auto-explicitation,
    title = "Bridging Background Knowledge Gaps in Translation with Automatic Explicitation",
    author = "Han, HyoJung  and Boyd-Graber, Jordan  and Carpuat, Marine",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://openreview.net/pdf?id=PBvSGqYCSa",
}
```