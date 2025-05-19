import torch
from typing import Dict
from .model import SmallLanguageModel
from .preprocessing import tokenize


def generate_sentence(
    model: SmallLanguageModel,
    start: str,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    word_to_class: Dict[str, str],
    idx_to_class: Dict[int, str],
    max_len: int = 20
) -> str:
    """
    Generiert einen Satz basierend auf einem Start-Text.

    :param model: Trainiertes Modell
    :param start: Start-Satz
    :return: Generierter Satz
    """
    model.eval()
    tokens = tokenize(start)
    input_ids = [word_to_idx.get(t, 0) for t in tokens]
    tensor = torch.tensor(input_ids).unsqueeze(0)
    hidden = model.init_hidden(1, model.rnn.hidden_size)
    output = tokens.copy()

    enders = {'.', '!', '?'}
    for _ in range(max_len - len(tokens)):
        wo, co, hidden = model(tensor, hidden)
        idx = torch.argmax(wo[:, -1]).item()
        word = idx_to_word[idx]
        output.append(word)
        if word in enders:
            break
        tensor = torch.tensor([[idx]])
    return ' '.join(output)