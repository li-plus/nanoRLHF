import torch

from train_rlhf import ByteTokenizer, GoldenRewardModel


def test_reward_model():
    tokenizer = ByteTokenizer()
    rm = GoldenRewardModel(tokenizer)
    inputs = tokenizer(
        [
            "goodbyeEgoodbyeE",
            "PPhelloEhelloEPP",
            "PPhelloEhalloEPP",
            "PPhelloEhelloooo",
            "PPPhellEsellllll",
        ],
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    reward_scores = rm(input_ids=input_ids, attention_mask=attention_mask)
    target_scores = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(reward_scores, target_scores)
