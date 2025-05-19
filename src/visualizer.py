import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa F401

def plot_top_words(words: list, counts: list) -> None:
    """
    Plottet die häufigsten Wörter als horizontales Balkendiagramm.

    :param words: Liste der Wörter
    :param counts: Liste der Häufigkeiten
    """
    plt.figure(figsize=(10, 5))
    plt.barh(words, counts)
    plt.gca().invert_yaxis()
    plt.xlabel('Häufigkeit')
    plt.title('Top-Wörter')
    plt.show()

def plot_class_distribution(class_counts: dict) -> None:
    """
    Plottet die Verteilung der Wortarten im Vokabular.

    :param class_counts: Dict mit Wortart -> Häufigkeit
    """
    classes, freqs = zip(*class_counts.items())
    plt.figure(figsize=(10, 5))
    plt.barh(classes, freqs)
    plt.gca().invert_yaxis()
    plt.xlabel('Anzahl')
    plt.title('Wortart-Verteilung')
    plt.show()

def plot_loss(history: dict) -> None:
    """
    Plottet den durchschnittlichen Gesamt-Loss pro Epoche.

    :param history: Dict mit den Keys 'epoch' und 'loss'
    """
    plt.figure(figsize=(12, 5))
    plt.plot(history['epoch'], history['loss'], marker='o', linestyle='-')
    plt.title('Verlauf des Gesamt-Loss pro Epoche')
    plt.xlabel('Epoche')
    plt.ylabel('Gesamt-Loss')
    plt.grid(True)
    plt.show()


def plot_embeddings(embeddings: np.ndarray, labels: list) -> None:
    """
    Zeichnet vier 3D-Scatterplots für zufällig ausgewählte Embedding-Dimensionen.

    :param embeddings: Array [n_words, embedding_dim]
    :param labels: Liste der Wort-Labels (gleiche Länge wie embeddings)
    """
    embedding_dim = embeddings.shape[1]
    np.random.seed(63)
    selected_dims = np.random.choice(embedding_dim, size=(4, 3), replace=False)

    fig = plt.figure(figsize=(12, 12))
    for i, (d1, d2, d3) in enumerate(selected_dims):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.scatter(
            embeddings[:, d1],
            embeddings[:, d2],
            embeddings[:, d3],
            alpha=0.5
        )
        for idx, lab in enumerate(labels):
            ax.text(
                embeddings[idx, d1],
                embeddings[idx, d2],
                embeddings[idx, d3],
                lab,
                fontsize=8
            )
        ax.set_title(f'Dim {d1} vs {d2} vs {d3}')
        ax.set_xlabel(f'Dimension {d1}')
        ax.set_ylabel(f'Dimension {d2}')
        ax.set_zlabel(f'Dimension {d3}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
