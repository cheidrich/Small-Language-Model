import random
import torch
import torch.nn as nn
from typing import List, Tuple
from .model import SmallLanguageModel
from .preprocessing import tokenize



vocab_size = 8000

epochs = 30
learning_rate = 0.005
word_loss_weight = 0.4
class_loss_weight = 0.6

embedding_dim = 25
hidden_dim = 100

max_generate_len = 20


def prepare_data(sentence: str, word_to_idx: dict, class_to_idx: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = tokenize(sentence)
    if len(tokens) < 2:
        return None, None, None
    cut = random.randint(1, len(tokens) - 1)
    inp = torch.tensor([word_to_idx.get(t, 0) for t in tokens[:cut]])
    tw = torch.tensor([word_to_idx.get(tokens[cut], 0)])
    tc = torch.tensor([class_to_idx.get(tokens[cut], 0)])
    return inp, tw, tc


def train(
    model: SmallLanguageModel,
    sentences: List[str],
    word_to_idx: dict,
    class_to_idx: dict
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    crit_w = nn.CrossEntropyLoss()
    crit_c = nn.CrossEntropyLoss()
    history = {'epoch': [], 'loss': [], 'lw': [], 'lc': []}

    for e in range(1, epochs + 1):
        total = 0.0
        history['epoch'].append(e)
        for s in sentences:
            inp, tw, tc = prepare_data(s, word_to_idx, class_to_idx)
            if inp is None:
                continue
            h0 = model.init_hidden(1, model.rnn.hidden_size)
            optimizer.zero_grad()
            wo, co, _ = model(inp.unsqueeze(0), h0)
            lw = crit_w(wo[:, -1], tw)
            lc = crit_c(co[:, -1], tc)
            history['lw'].append(lw.item())
            history['lc'].append(lc.item())
            loss = word_loss_weight * lw + class_loss_weight * lc
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(sentences)
        history['loss'].append(avg)
        print(f"Epoche {e}/{epochs} abgeschlossen: Loss={avg:.4f}")

    return history

if __name__ == '__main__':
    # Beispiel-Aufruf: train(model, sentences, w2i, c2i)
    pass