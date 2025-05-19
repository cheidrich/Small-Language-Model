from collections import Counter
from typing import Dict, List, Tuple
from .preprocessing import tokenize, get_word_class


def build_vocab(sentences: List[str], max_size: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Erstellt ein Vokabular der häufigsten Tokens.

    :param sentences: Liste gereinigter Sätze
    :param max_size: Maximale Vokabulargröße
    :return: (vocab list, word->idx, idx->word)
    """
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(tokenize(sent))
    counts = Counter(all_tokens)
    vocab = [w for w, _ in counts.most_common(max_size)]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w


def assign_classes(vocab: List[str]) -> Tuple[Dict[str, str], List[str], Dict[str, int], Dict[int, str]]:
    """
    Weist jedem Token eine Wortart zu.

    :param vocab: Liste der Tokens
    :return: (word->class, liste der Klassen, class->idx, idx->class)
    """
    w2c = {w: get_word_class(w) for w in vocab}
    classes = sorted(set(w2c.values()))
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    return w2c, classes, c2i, i2c