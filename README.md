# nanoRLHF

Train a tiny LLaMA model from scratch to repeat your words using Reinforcement Learning from Human Feedback ([RLHF](https://huggingface.co/blog/rlhf)).

This is a tiny working demo to train a language model using PPO algorithm. In this task, the dataset contains ~50k common words in web corpus. Each word serves as a sample. A byte tokenizer is applied to encode each letter in the word into a token. The reward model here is a golden rule that gives higher scores to longer prefix match between prompt and response. The policy model is trained from scratch to maximize its rewards. Gradually, it learns to repeat the prompt letter by letter.

## Quick Start

Install necessary dependencies:
```sh
pip install torch transformers wandb nltk tabulate
```

Download the word list as training data. Start a Python interpreter and type:
```python
>>> import nltk
>>> nltk.download("words")
```

Start training on the word list:
```sh
python3 train_rlhf.py
```

If the training goes well, the final validation accuracy should reach 100%.

Start the interactive demo to load the checkpoint and chat with it.
```
$ python3 chat_rlhf.py
Please type a single word in lower case within 7 letters at one time. For example, type "hello" and press enter.
nanoRLHF > hello
hello
nanoRLHF > nano
nano
nanoRLHF > rlhf
rlhf
```

Note that "rlhf" is not on the word list. The model is capable to generalize its abilities to unseen words.

## Acknowledgements

We have learned a lot from the open source community and we appreciate the below projects:
* [huggingface/trl](https://github.com/huggingface/trl): most of our PPO implementation is adapted from trl.
* [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat): the training pipeline are adapted from DS-Chat, then made even simpler.
