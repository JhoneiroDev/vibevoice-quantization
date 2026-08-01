import torch

from vibevoice.finetune.train_vibevoice import mask_for_ce


def test_mask_for_ce_keeps_target_continuations_and_terminal_token() -> None:
    labels = torch.tensor([[10, 20, 30, 30, 40, 99]])
    attention_mask = torch.ones_like(labels)
    acoustic_loss_mask = torch.tensor([[False, False, True, True, False, False]])

    masked = mask_for_ce(labels, attention_mask, acoustic_loss_mask)

    assert masked.tolist() == [[-100, 30, 30, 40, -100]]


def test_mask_for_ce_ignores_padding_and_prompt_audio() -> None:
    labels = torch.tensor([[10, 30, 20, 30, 40, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    acoustic_loss_mask = torch.tensor([[False, False, False, True, False, False]])

    masked = mask_for_ce(labels, attention_mask, acoustic_loss_mask)

    assert masked.tolist() == [[-100, -100, 30, 40, -100]]
