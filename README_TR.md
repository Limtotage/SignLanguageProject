# Türk İşaret Dili Çeviri Programı

[🇬🇧 English README](README.md)

Kamera üzerinden Türk İşaret Dili (TİD) işaretlerini algılayan, MediaPipe ile landmark çıkaran, GRU tabanlı bir sinir ağı ile kelime tanıyan ve tanınan kelimeleri NLP/sentence engine üzerinden Türkçe cümlelere dönüştüren sistem.

## Mevcut Sistem

Mevcut işlem hattı:

```text
Kamera
   ↓
MediaPipe
   ↓
Landmark Çıkarma
   ↓
Normalizasyon
   ↓
90 Karelik Dizi
   ↓
GRU Modeli
   ↓
Tahmin Stabilizasyonu
   ↓
Tanınan Kelimeler
   ↓
NLP Parser
   ↓
Sentence Engine
   ↓
Türkçe Cümle
```

Sistem şu anda **20 gerçek kelime + DEFAULT** olmak üzere toplam **21 sınıf** tanımaktadır.

---

## Tanınan Kelimeler

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

`DEFAULT`, hedef kelimelerden birinin işaret edilmediği nötr/dinlenme durumunu temsil eder.

---

## Proje Yapısı

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

## Landmark Çıkarma

Sistem MediaPipe Tasks kullanarak:

* El landmarkları
* Vücut/pose landmarkları
* Yüz landmarkları

çıkarmaktadır.

Her karede toplam **109 landmark** bulunur ve her landmark 3 koordinattan oluşur:

```text
109 × 3 = 327 özellik/kare
```

Dağılım:

```text
Sol El       → 21 landmark
Sağ El       → 21 landmark
Pose         → 33 landmark
Yüz          → 34 seçilmiş landmark
----------------------------------
Toplam       → 109 landmark
```

Yüz için MediaPipe'ın tüm landmarkları kullanılmaz. Sistemde seçilmiş 34 yüz landmarkı kullanılmaktadır.

---

## Normalizasyon

Landmark koordinatları omuz bölgesine göre normalize edilir.

Omuz merkezi şu pose noktaları kullanılarak hesaplanır:

```text
points[53]
points[54]
```

Ardından koordinatlar omuzlar arasındaki mesafeye göre ölçeklendirilir.

Bu işlem şu farklılıkların etkisini azaltır:

* Kameraya olan uzaklık
* Kişinin kamera içindeki konumu
* Vücut boyutu
* Kameradaki küçük konum değişiklikleri

---

## Dizi İşleme

GRU modeli sabit uzunlukta zamansal diziler kullanır.

```text
Dizi uzunluğu = 90 kare
Kare başına özellik = 327
```

Dolayısıyla model girdisinin şekli:

```text
(90, 327)
```

Dataset hazırlama aşamasında diziler 90 kareye yeniden örneklenir ve:

```text
processed/X.npy
processed/y.npy
```

dosyaları oluşturulur.

---

## GRU Modeli

İşaret tanıma modeli GRU tabanlıdır.

Mevcut mimari:

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

Eğitim ayarları:

```text
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
Epochs: 100
Batch size: 8
Validation split: %20
Random state: 42
Early stopping patience: 10
En iyi ağırlıkları geri yükleme: Aktif
```

Final modelinin tam olarak **21 sınıf** üretmesi gerekir. Bu sınıfların sırası `LABELS` listesi ve dataset sınıf eşleştirmesiyle aynı olmalıdır.

---

## Tahmin Stabilizasyonu

Canlı sistem modelin yaptığı her tahmini doğrudan kelime olarak kabul etmez.

Mevcut ayarlar:

```text
SEQUENCE_LENGTH = 90
PREDICT_EVERY = 5

CONFIDENCE_THRESHOLD = 0.70
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

STABLE_PREDICTIONS = 5
WORD_STABLE_COUNT = 4
DEFAULT_STABLE_COUNT = 3
```

Normal bir kelimenin kabul edilmesi için son tahminlerin yeterince stabil olması gerekir.

Örneğin son 5 tahminin en az 4 tanesi aynı kelimeyse ve güven değeri yeterliyse kelime kabul edilir.

Kelime kabul edildiğinde sistem aynı işaretin arka arkaya tekrar tekrar eklenmesini engellemek için kilitlenir.

Yeni bir kelime kabul edilmeden önce belirli sayıda stabil `DEFAULT` tahmini beklenir.

Bu yapı tahminlerdeki dalgalanmayı ve aynı kelimenin yanlışlıkla birden fazla eklenmesini azaltır.

---

## NLP Sistemi

NLP sistemi:

```text
src/nlp_engine.py
```

dosyasında bulunmaktadır.

Sistem tanınan kelimelere dilbilgisel roller atayan bir sözlük/lexicon kullanmaktadır.

Örnekler:

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

Parser şu bilgileri çıkarabilmektedir:

```text
Özne
Nesne
Fiil
Zaman
Soru
Sıfatlar
Cümle tipi
```

Parser kelimelerin tam olarak hangi sırada geldiğine bağımlı değildir.

Örneğin:

```text
BEN BUGUN GELMEK
BUGUN BEN GELMEK
```

ifadelerinin ikisi de:

```text
Özne → BEN
Fiil → GELMEK
Zaman → BUGUN
```

olarak yorumlanır.

---

## Sentence Engine

Cümle oluşturma sistemi:

```text
src/sentence_engine.py
```

dosyasında bulunmaktadır.

Sentence engine artık yalnızca sabit kelime dizilerine bağlı çalışmak yerine NLP parser'ın çıkardığı rolleri kullanmaktadır.

Şu örnekler çalışmaktadır:

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

## Canlı Tanıma

Canlı sistem:

```text
src/live_predict.py
```

dosyasında bulunmaktadır.

Sistemin canlı işlem sırası:

```text
Kamera
→ MediaPipe
→ Landmark normalizasyonu
→ GRU
→ Tahmin stabilizasyonu
→ Kelime dizisi
→ NLP
→ Cümle oluşturma
```

Tanınan kelimeler:

```python
sentence_words
```

listesinde tutulur.

Stabil bir kelime kabul edildiğinde kelime dizisi ve oluşturulan cümle terminalde gösterilir.

Örnek:

```text
KELİME EKLENDİ: GELMEK
KELİME DİZİSİ: ['BEN', 'BUGUN', 'GELMEK']
CÜMLE: Ben bugün geliyorum.
```

---

## Dataset

Dataset kelime/sınıf klasörlerine ayrılmıştır:

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

Her işaret dizisi landmark koordinatlarını içeren NumPy verisi olarak saklanmaktadır.

---

## Test

NLP ve sentence engine testleri şu komutla çalıştırılabilir:

```bash
python3 src/test_nlp.py
```

Testlerde şu özellikler kontrol edilmektedir:

* Zamirler
* Özne/nesne tespiti
* Fiil tespiti
* Zaman ifadeleri
* Sorular
* İsimler
* Sıfatlar
* Selamlaşmalar
* Cevaplar
* Cümle oluşturma

---

## Mevcut Uçtan Uca Sonuç

Sistemin tamamı gerçek kamera görüntüsüyle test edilmiştir.

Başarılı bir örnek:

```text
Tanınan kelimeler:
BEN
BUGUN
GELMEK

Oluşturulan cümle:
Ben bugün geliyorum.
```

Bu sonuç, mevcut sistemin kamera görüntüsünden başlayarak işareti tanıyıp NLP katmanından geçirerek Türkçe cümle oluşturabildiğini göstermektedir.

---

