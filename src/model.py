import torch
import torch.nn as nn
from typing import Tuple


class SmallLanguageModel(nn.Module):
    """
    Einfaches RNN-Sprachmodell.

    :param vocab_size: Größe des Vokabulars
    :param class_size: Anzahl der Wortklassen
    :param embedding_dim: Embedding-Dimension
    :param hidden_dim: Hidden-State-Dimension
    """
    def __init__(self, vocab_size: int, class_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc_word = nn.Linear(hidden_dim, vocab_size)
        self.fc_class = nn.Linear(hidden_dim, class_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        return self.fc_word(output), self.fc_class(output), hidden

    def init_hidden(self, batch_size: int, hidden_dim: int) -> torch.Tensor:
        return torch.zeros(1, batch_size, hidden_dim)