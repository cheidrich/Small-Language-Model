import re
from typing import List, Tuple

# Wortart- und Zeichen-Sets
SATZZEICHEN = {",", ".", ":", "?"}
ARTIKEL = {"der", "die", "das", "ein", "eine"}
PRÄPOSITIONEN = {"in", "auf", "unter", "mit", "vor"}
KONJUNKTIONEN = {"aber", "und", "oder", "dass"}


def tokenize(sentence: str) -> List[str]:
    """
    Zerlegt einen Satz in Tokens (Wörter und Satzzeichen).

    :param sentence: Eingabesatz als String
    :return: Liste von Tokens (Wort-Strings)
    """
    sentence = re.sub(r'["“”»«]', '', sentence)
    tokens = re.findall(r'\w+|[^\w\s]', sentence)
    return [tok for tok in tokens if tok in SATZZEICHEN or tok.isalnum()]


def get_word_class(word: str, is_first_in_sentence: bool = False) -> str:
    """
    Bestimmt die Wortart eines Wortes.

    :param word: Token als String
    :param is_first_in_sentence: Ob das Wort am Satzanfang steht
    :return: Wortart als String
    """
    if word in ARTIKEL:
        return "Artikel"
    elif word in PRÄPOSITIONEN:
        return "Präposition"
    elif word in KONJUNKTIONEN:
        return "Konjunktion"
    elif word in SATZZEICHEN:
        return "Satzzeichen"
    elif word[0].isupper() and not is_first_in_sentence:
        return "Nomen"
    elif word.endswith("en") and not word[0].isupper():
        return "Verb"
    else:
        return "Adjektiv"


def load_sentences(path: str) -> List[str]:
    """
    Liest Rohdaten und gibt die Nachrichtentexte zurück.

    :param path: Pfad zur Textdatei
    :return: Liste von Sätzen
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split('\t')[1] for line in f]


def clean_and_filter(sentences: List[str], vocab_set: set) -> Tuple[List[str], List[str]]:
    """
    Tokenisiert, säubert und filtert Sätze nach Vokabular.

    :param sentences: Liste der Roh-Sätze
    :param vocab_set: Menge erlaubter Tokens
    :return: Tuple (gereinigte Sätze, gefilterte Sätze)
    """
    cleaned, filtered = [], []
    for sent in sentences:
        tokens = tokenize(sent)
        if tokens:
            joined = " ".join(tokens)
            cleaned.append(joined)
            if all(tok in vocab_set for tok in tokens):
                filtered.append(joined)
    return cleaned, filtered