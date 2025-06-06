# Small-Language-Model

In diesem Repository möchte ich sowohl selbst-trainierte als auch vortrainierte Sprachmodelle aus Hugging Face testen. Das Endresultat sieht man hier in Form der Embeddings bzw. des semantischen Vektorraums. Hier werden Wörter näher anneinander gecluster, je mehr sie semantisch in der Trainingsmenge miteinander zu tun haben.

![Embeddings](eda-notebooks/semantic-vector-space.png)

---

## Self-Trained

Download:
https://wortschatz.uni-leipzig.de/de/download/German

Dateien für Training:

- `deu_news_2024_10K-words.txt`
- `deu_news_2024_10K-sentences.txt`

## Beschreibung des Modells und der Architektur

Dieses Projekt implementiert ein kompaktes RNN-basiertes Sprachmodell, das auf deutschen Nachrichtensätzen trainiert wird. Ziel ist es, sowohl das nächste Wort als auch dessen Wortart vorherzusagen. Die Architektur und den Ablauf wurden modular aufgebaut:

### 1. Datenaufbereitung
- **Datenquelle**  
  Die Datei `deu_news_2024_10K-sentences.txt` enthält 10.000 realistische Nachrichtensätze.
- **Tokenisierung**  
  Mit `re.findall(r'\w+|[^\w\s]', sentence)` werden Wörter und Satzzeichen extrahiert. Andere Zeichen werden verworfen, um das Vokabular überschaubar zu halten.
- **Wortarten-Klassifikation**  
  Harte Regeln (Artikel, Präpositionen, Konjunktionen, Satzzeichen) und Heuristiken (Großschreibung, Wortendungen) teilen jedes Token einer von fünf Klassen zu.
- **Vokabular**  
  Die häufigsten 8.000 Tokens bilden das Grundvokabular. Alle Sätze werden bereinigt und auf dieses Vokabular gefiltert.
- **Datenaugmentation**  
  Häufige Sätze werden mehrfach in die Trainingsdaten aufgenommen, um seltene Fehler durch seltene Wörter zu reduzieren.

### 2. Modellarchitektur
- **Embedding‑Schicht**  
  `nn.Embedding(vocab_size, embedding_dim)` transformiert Token-Indizes in dichte Vektoren (Embedding-Dimension z. B. 25).
- **RNN‑Schicht**  
  `nn.RNN(embedding_dim, hidden_dim, batch_first=True)` modelliert die Sequenzdynamik (Hidden-Dimension z. B. 100).
- **Ausgabeschichten**  
  - `fc_word = nn.Linear(hidden_dim, vocab_size)` zur Wortvorhersage  
  - `fc_class = nn.Linear(hidden_dim, class_size)` zur Wortartvorhersage  
- **Forward-Pass**  
  Eingabe → Embedding → RNN → zwei parallele Klassifikatoren für Token und Wortart.

### 3. Training
- **Beispiel-Erstellung**  
  Für jeden Satz wählt `prepare_data` zufällig einen „Cut-Point“:  
  - Wörter vor der Trennstelle sind Input  
  - Das Token an der Trennstelle ist Ziel für Wort- und Wortartvorhersage
- **Loss-Funktionen**  
  - `criterion_word = CrossEntropyLoss()` für Wort  
  - `criterion_class = CrossEntropyLoss()` für Wortart  
  - Gesamt-Loss = `α * loss_word + (1–α) * loss_class` (Gewichtung über Config steuerbar)
- **Optimierung**  
  `torch.optim.Adam` mit konfigurierbarem Lernrate-Parameter.
- **Verlauf**  
  Pro Epoche werden Durchschnitts‑Loss, Wort‑Loss und Wortart‑Loss aufgezeichnet und später visualisiert.
- **Persistenz**  
  Nach dem Training speichert das Skript das Modell‑State‑Dictionary als `language_model.pth`.

### 4. Visualisierung
- **Top-Wörter**  
  Horizontale Balkendiagramme der häufigsten Tokens.
- **Wortart-Verteilung**  
  Balkendiagramme, die Anzahl der Tokens je Klasse zeigen.
- **Loss-Verlauf**  
  Zwei Plots pro Epoche
  - Gesamt-Loss  
  - Komponenten: Wort-Loss, Wortart-Loss und Gesamt-Loss  
- **Embedding‑Projektion**  
  Zufällige Auswahl von drei Dimensions‑Tripeln für 3D-Scatterplots, um semantische Strukturen im Vektorraum zu erkunden.

### 5. Generierung
- **Greedy Sampling**  
  Zu Demonstrationszwecken wird aktuell Argmax verwendet; zukünftige Versionen könnten Temperatur‑Sampling einsetzen.
- **Großschreibung & Satzzeichen**  
  Das erste Wort und alle als Nomen klassifizierten Tokens werden großgeschrieben. Satzzeichen werden korrekt ohne Leerzeichen vorangestellt.

---

## Pre-Trained

### Apotheken-Chatbot

Dieses Repository enthält außerdem ein Jupyter Notebook, das einen Apotheken-Chatbot implementiert (unter Verwendung eines vortrainierten Modells `google/flan-t5-small` von HuggingFace). Der Chatbot beantwortet Fragen zu Medikamenten, einschließlich Kosten, Verfügbarkeit, Anwendungsgebieten und Dosierung, basierend auf einer vordefinierten Apotheken-Datenbank. Natürlich ist es wichtig, dass ein solcher Chatbot korrekte Antworten liefert und Aussagen der Nutzer kritisch hinterfragt.

#### Hauptfunktionen:
- **Antworten:** Generiert Antworten aus der Datenbank und strukturiert die Prompts für präzise und verständliche Antworten.  
- **Sicherheitsprüfungen:** Überprüft die Anwendungsgebiete von Medikamenten, um fehlerhafte Nutzungsempfehlungen zu vermeiden.  
- **Rückfragen:** Schlägt automatisch relevante Rückfragen vor (z. B. zur Dosierung), basierend auf den Nutzeranfragen.  
