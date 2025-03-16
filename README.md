# Small-Language-Model

Download:
https://wortschatz.uni-leipzig.de/de/download/German

Dateien für Training:

deu_news_2024_10K-words.txt
deu_news_2024_10K-sentences.txt

## Beschreibung des Modells und der Vorarbeit

Dieses Projekt umfasst ein Sprachmodell, das auf deutschen Nachrichtensätzen trainiert wird, um das nächste Wort und dessen Wortart vorherzusagen. Es wurde schrittweise entwickelt und angepasst, um Overfitting zu minimieren, die Großschreibung von Nomen zu gewährleisten, und eine flexible Visualisierung zu bieten. Hier ist eine detaillierte Übersicht:

### 1. Datenaufbereitung
- **Datenquelle**: Die Datei `deu_news_2024_10K-sentences.txt` enthält 10.000 Sätze, die eingelesen und verarbeitet werden.
- **Tokenisierung**: 
  - Funktion `tokenize`: Zerlegt Sätze in Wörter und Satzzeichen mit `re.findall(r'\w+|[^\w\s]', sentence)`.
  - Alle Anführungszeichen (`"`, `“`, `”`, `»`, `«`) werden entfernt (`re.sub(r'["“”»«]', '', sentence)`).
  - Nur bestimmte Satzzeichen (`,`, `.`, `:`, `?`) werden beibehalten; andere werden durch leere Strings ersetzt.
  - Keine `.lower()`-Konvertierung, um die ursprüngliche Groß-/Kleinschreibung zu erhalten.
- **Wortarten**:
  - Hardgecodete Listen: `ARTIKEL`, `PRÄPOSITIONEN`, `KONJUNKTIONEN`, `SATZZEICHEN`.
  - Funktion `get_word_class`: Klassifiziert Wörter heuristisch:
    - Artikel, Präpositionen, Konjunktionen: Direkt zugeordnet.
    - Satzzeichen: Als „Satzzeichen“ klassifiziert.
    - Nomen: Großgeschrieben, außer am Satzanfang.
    - Verben: Enden auf „en“ und nicht großgeschrieben.
    - Adjektive: Fallback für alles andere.
- **Grundvokabular**: Die 10.000 häufigsten Wörter (`grundvokabular`) werden bestimmt, inklusive ihrer Wortarten (`word_to_class`).
- **Filterung**: Sätze werden bereinigt (`cleaned_sentences`) und gefiltert (`filtered_sentences`), sodass nur solche mit Wörtern aus dem Grundvokabular übrigbleiben.
- **Trainingsmenge**:
  - Sätze mit den 200 häufigsten Wörtern werden verzehnfacht.
  - Sätze mit den 850 häufigsten Wörtern werden verdreifacht.
  - Andere Sätze bleiben unverändert.

### 2. Modellarchitektur
- **Klasse**: `ImprovedLanguageModel` (erbt von `nn.Module`).
- **Komponenten**:
  - **Embedding-Schicht**: `nn.Embedding(vocab_size, embedding_dim)` mit `embedding_dim = 50` (klein gewählt gegen Overfitting).
  - **RNN-Schicht**: `nn.RNN(embedding_dim, hidden_dim, batch_first=True)` mit `hidden_dim = 100`.
  - **Ausgabeschichten**:
    - `fc_word`: `nn.Linear(hidden_dim, vocab_size)` für Wortvorhersagen.
    - `fc_class`: `nn.Linear(hidden_dim, class_size)` für Wortartvorhersagen.
- **Forward-Pass**: Gibt `word_output`, `class_output`, und `hidden` zurück.

### 3. Training
- **Datenaufbereitung**:
  - Funktion `prepare_data`: 
    - Wählt einen zufälligen Schnittpunkt (`cut_point`) im Satz.
    - `inputs`: Wörter vor dem Schnittpunkt.
    - `targets_word`: Das Wort am Schnittpunkt.
    - `targets_class`: Die Wortart des Ziel-Worts.
- **Loss**:
  - `criterion_word`: `nn.CrossEntropyLoss()` für Wortvorhersagen.
  - `criterion_class`: `nn.CrossEntropyLoss()` für Wortartvorhersagen.
  - Gesamt-Loss: `loss = 0.7 * loss_word + 0.3 * loss_class` (Gewichtung anpassbar).
- **Optimierung**: `torch.optim.Adam` mit `lr=0.01`.
- **Schleife**: 7 Epochen, Loss wird alle 20 Sätze aufgezeichnet (`loss_history`, `wl_history`, `cl_history`).
- **Speicherung**: Nach dem Training als `state_dict` in `"language_model.pth"`.

### 4. Generierung
- **Funktion**: `generate_sentence`:
  - Start-Satz wird tokenisiert, Eingabe wird schrittweise erweitert.
  - Sampling mit Temperatur (`temperature=1.2`) statt `argmax` für Vielfalt.
  - Großschreibung:
    - Erstes Wort: Immer groß.
    - Nomen: Immer groß (basierend auf `word_to_class` oder Modellvorhersage).
  - Satzzeichen: Kein Leerzeichen davor.

### 5. Visualisierung
- **Häufigste Wörter**: Waagrechter Balkenplot der 7 häufigsten Wörter in Himmelblau (`skyblue`).
- **Wortarten**: Waagrechter Balkenplot der Wortarten-Verteilung in Hellgrün (`lightgreen`).
- **Loss-Verlauf**: Linienplot mit:
  - Gesamt-Loss (Blau), Wort-Loss (Rot), Wortart-Loss (Grün).
  - Alle 20 Sätze aufgezeichnet.

### Besondere Anpassungen
- **Overfitting**: Kleine `embedding_dim = 50` gegen Overfitting bei begrenztem Datensatz.
- **Satzzeichen**: Nur `,`, `.`, `:`, `?` erlaubt, Anführungszeichen entfernt.
- **Zufälliges Training**: Sätze werden zufällig abgeschnitten, um Generalisierung zu fördern.

### Abschließende Bemerkungen
Dieses Modell ist ein kompaktes RNN-basiertes Sprachmodell, optimiert für deutsche Sätze mit einem Fokus auf Großschreibung von Nomen und minimalistischem Satzzeichen-Handling. Es ist gegen Overfitting durch eine kleine `embedding_dim` geschützt und bietet eine detaillierte Visualisierung der Trainings- und Datenmetriken.


