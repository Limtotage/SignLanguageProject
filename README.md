# Turkish Sign Language Translation Program

[🇹🇷 Türkçe README](README_TR.md)

A Turkish Sign Language (TİD) recognition system that uses a camera, MediaPipe landmark extraction, a GRU-based neural network, and a rule-based NLP/sentence engine to convert recognized signs into Turkish sentences.

## Current System

The current pipeline is:

```text
Camera
   ↓
MediaPipe
   ↓
Landmark Extraction
   ↓
Normalization
   ↓
90-Frame Sequence
   ↓
GRU Model
   ↓
Prediction Stabilization
   ↓
Recognized Words
   ↓
NLP Parser
   ↓
Sentence Engine
   ↓
Turkish Sentence
```

The system currently recognizes **20 Turkish words + DEFAULT**, for a total of **21 model classes**.

---

## Recognized Vocabulary

```text
DEFAULT
BEN
SEN
SEVMEK
MERHABA
TESEKKUR
EVET
HAYIR
GELMEK
GITMEK
YARDIM
NE
NEREDE
SU
YEMEK
ARKADAS
BUGUN
YARIN
IYI
KOTU
TEKRAR
```

`DEFAULT` represents the neutral/rest state when no target sign is being recognized.

---

## Project Structure

```text
SignLanguageProject/
├── dataset/
├── processed/
│   ├── X.npy
│   └── y.npy
├── models/
│   ├── tid_gru.keras
│   ├── hand_landmarker.task
│   ├── face_landmarker.task
│   └── pose_landmarker_full.task
└── src/
    ├── prepare_dataset.py
    ├── train_gru.py
    ├── live_predict.py
    ├── sentence_engine.py
    ├── nlp_engine.py
    └── test_nlp.py
```

---

## Landmark Extraction

The system uses MediaPipe Tasks for:

* Hand landmarks
* Pose landmarks
* Face landmarks

Each frame contains **109 landmarks**, with 3 coordinates per landmark:

```text
109 × 3 = 327 features per frame
```

The landmark structure is:

```text
Left Hand   → 21 landmarks
Right Hand  → 21 landmarks
Pose        → 33 landmarks
Face        → 34 selected landmarks
-------------------------------------
Total       → 109 landmarks
```

The face model does not use every available face landmark. A selected set of 34 landmarks is used.

---

## Normalization

The landmark coordinates are normalized around the shoulder area.

The shoulder center is calculated using pose landmarks:

```text
points[53]
points[54]
```

The coordinates are then scaled using the shoulder distance.

This reduces the effect of differences in:

* Camera distance
* Person position
* Body size
* Small movements in the camera frame

---

## Sequence Processing

The model works with fixed-length temporal sequences.

```text
Sequence length = 90 frames
Features/frame = 327
```

Therefore, each model input has the shape:

```text
(90, 327)
```

The dataset preparation process resamples sequences to 90 frames and produces:

```text
processed/X.npy
processed/y.npy
```

---

## GRU Model

The recognition model is based on a GRU neural network.

Current architecture:

```python
Sequential([
    GRU(128, return_sequences=True),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])
```

Training configuration:

```text
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
Epochs: 100
Batch size: 8
Validation split: 20%
Random state: 42
Early stopping patience: 10
Restore best weights: enabled
```

The final model must produce exactly **21 output classes**, matching the `LABELS` list and dataset class mapping.

---

## Prediction Stabilization

The live prediction system does not immediately accept every model prediction.

Current settings:

```text
SEQUENCE_LENGTH = 90
PREDICT_EVERY = 5

CONFIDENCE_THRESHOLD = 0.70
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

STABLE_PREDICTIONS = 5
WORD_STABLE_COUNT = 4
DEFAULT_STABLE_COUNT = 3
```

A normal word is accepted when the recent predictions are sufficiently stable.

For example, if the last 5 predictions contain the same word at least 4 times and the confidence is high enough, the word is accepted.

After a word is accepted, the system locks the word to prevent the same sign from being inserted repeatedly.

The system waits for several stable `DEFAULT` predictions before allowing another word to be accepted.

This significantly reduces duplicate word insertion and prediction noise.

---

## NLP System

The project currently contains an NLP layer in:

```text
src/nlp_engine.py
```

The NLP system contains a vocabulary/lexicon that assigns roles to recognized words.

Examples:

```text
BEN        → PRONOUN
SEN        → PRONOUN
SEVMEK     → VERB
GELMEK     → VERB
GITMEK     → VERB
YEMEK      → AMBIGUOUS
SU         → NOUN
ARKADAS    → NOUN
YARDIM     → NOUN
BUGUN      → TIME
YARIN      → TIME
NE         → QUESTION
NEREDE     → QUESTION
IYI        → ADJECTIVE
KOTU       → ADJECTIVE
MERHABA    → GREETING
TESEKKUR   → GREETING
EVET       → ANSWER
HAYIR      → ANSWER
TEKRAR     → CONTROL
```

The parser extracts information such as:

```text
Subject
Object
Verb
Time
Question
Adjectives
Sentence type
```

The parser is not dependent on the exact order of the recognized words.

For example:

```text
BEN BUGUN GELMEK
BUGUN BEN GELMEK
```

are both interpreted as:

```text
Subject → BEN
Verb    → GELMEK
Time    → BUGUN
```

---

## Sentence Engine

The sentence generation system is implemented in:

```text
src/sentence_engine.py
```

It uses the NLP parser instead of relying only on fixed word sequences.

Examples currently working:

```text
['BEN', 'SEN', 'SEVMEK']
→ Ben seni seviyorum.

['SEN', 'BEN', 'SEVMEK']
→ Sen beni seviyorsun.

['BEN', 'BUGUN', 'GELMEK']
→ Ben bugün geliyorum.

['BUGUN', 'BEN', 'GELMEK']
→ Ben bugün geliyorum.

['SEN', 'NEREDE']
→ Sen neredesin?

['BEN', 'SU']
→ Ben su istiyorum.

['BEN', 'YEMEK']
→ Ben yemek yiyorum.

['BEN', 'ARKADAS']
→ Ben arkadaşım.

['BEN', 'IYI']
→ Ben iyiyim.

['BEN', 'KOTU']
→ Ben kötüyüm.

['MERHABA']
→ Merhaba.

['TESEKKUR']
→ Teşekkür ederim.

['EVET']
→ Evet.

['HAYIR']
→ Hayır.
```

---

## Live Recognition

The live system is implemented in:

```text
src/live_predict.py
```

It connects:

```text
Camera
→ MediaPipe
→ Landmark normalization
→ GRU
→ Prediction stabilization
→ Word sequence
→ NLP
→ Sentence generation
```

Recognized words are stored in:

```python
sentence_words
```

When a stable word is accepted, the current word sequence and generated sentence are displayed.

Example:

```text
KELİME EKLENDİ: GELMEK
KELİME DİZİSİ: ['BEN', 'BUGUN', 'GELMEK']
CÜMLE: Ben bugün geliyorum.
```

---

## Dataset

The dataset is organized by word/class folders:

```text
dataset/
├── default/
├── ben/
├── sen/
├── sevmek/
├── merhaba/
├── tesekkur/
├── evet/
├── hayir/
├── gelmek/
├── gitmek/
├── yardim/
├── ne/
├── nerede/
├── su/
├── yemek/
├── arkadas/
├── bugun/
├── yarin/
├── iyi/
├── kotu/
└── tekrar/
```

Each sequence is stored as NumPy data containing landmark coordinates.

---

## Testing

The NLP and sentence engine can be tested using:

```bash
python3 src/test_nlp.py
```

The test suite currently covers:

* Pronouns
* Subject/object detection
* Verb detection
* Time expressions
* Questions
* Nouns
* Adjectives
* Greetings
* Answers
* Sentence generation

---

## Current End-to-End Result

The complete pipeline has been tested with live camera input.

A successful example:

```text
Recognized words:
BEN
BUGUN
GELMEK

Generated sentence:
Ben bugün geliyorum.
```

This confirms that the current system can process a sign from camera input all the way through to a generated Turkish sentence.

---

