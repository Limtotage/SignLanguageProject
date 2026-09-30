# src/nlp_engine.py

LEXICON = {
    "BEN": {
        "role": "PRONOUN",
        "person": 1,
        "number": "singular"
    },

    "SEN": {
        "role": "PRONOUN",
        "person": 2,
        "number": "singular"
    },

    "SEVMEK": {
        "role": "VERB",
        "lemma": "sevmek"
    },

    "GELMEK": {
        "role": "VERB",
        "lemma": "gelmek"
    },

    "GITMEK": {
        "role": "VERB",
        "lemma": "gitmek"
    },

    "YEMEK": {
        "role": "AMBIGUOUS",
        "possible_roles": ["NOUN", "VERB"],
        "lemma": "yemek"
    },

    "SU": {
        "role": "NOUN"
    },

    "ARKADAS": {
        "role": "NOUN"
    },

    "YARDIM": {
        "role": "NOUN"
    },

    "BUGUN": {
        "role": "TIME"
    },

    "YARIN": {
        "role": "TIME"
    },

    "NE": {
        "role": "QUESTION"
    },

    "NEREDE": {
        "role": "QUESTION"
    },

    "IYI": {
        "role": "ADJECTIVE"
    },

    "KOTU": {
        "role": "ADJECTIVE"
    },

    "MERHABA": {
        "role": "GREETING"
    },

    "TESEKKUR": {
        "role": "GREETING"
    },

    "EVET": {
        "role": "ANSWER"
    },

    "HAYIR": {
        "role": "ANSWER"
    },

    "TEKRAR": {
        "role": "CONTROL"
    }
}


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize(words):
    """
    Modelden gelen kelime listesini temizler.
    """

    if not words:
        return []

    tokens = []

    for word in words:

        if not word:
            continue

        word = word.upper().strip()

        if word == "DEFAULT":
            continue

        if word in LEXICON:
            tokens.append(word)

    return tokens


# --------------------------------------------------
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------

def find_subject(tokens):
    """
    İlk uygun zamiri özne olarak belirler.
    """

    for token in tokens:

        info = LEXICON.get(token)

        if not info:
            continue

        if info["role"] == "PRONOUN":
            return token

    return None


def find_verb(tokens):
    """
    Gerçek fiili bulur.

    YEMEK özel durumdur:
    Eğer YEMEK'ten önce bir özne varsa,
    bu aşamada fiil olarak kabul edilir.
    """

    for token in tokens:

        info = LEXICON.get(token)

        if not info:
            continue

        if info["role"] == "VERB":
            return token

        if token == "YEMEK":
            return "YEMEK"

    return None


def find_time(tokens):
    """
    Zaman ifadesini bulur.
    """

    for token in tokens:

        info = LEXICON.get(token)

        if info and info["role"] == "TIME":
            return token

    return None


def find_question(tokens):
    """
    Soru kelimesini bulur.
    """

    for token in tokens:

        info = LEXICON.get(token)

        if info and info["role"] == "QUESTION":
            return token

    return None


def find_adjectives(tokens):
    """
    Cümledeki sıfatları bulur.
    """

    adjectives = []

    for token in tokens:

        info = LEXICON.get(token)

        if info and info["role"] == "ADJECTIVE":
            adjectives.append(token)

    return adjectives


def find_nouns(tokens, subject=None, verb=None):
    """
    Özne ve fiil olmayan isimleri bulur.
    """

    nouns = []

    for token in tokens:

        if token == subject:
            continue

        if token == verb:
            continue

        info = LEXICON.get(token)

        if not info:
            continue

        if info["role"] == "NOUN":
            nouns.append(token)

    return nouns


# --------------------------------------------------
# OBJECT ANALYSIS
# --------------------------------------------------

def find_object(tokens, subject=None, verb=None):
    """
    Nesneyi belirler.

    Örneğin:

    BEN SEN SEVMEK
    ↓
    subject = BEN
    object = SEN
    verb = SEVMEK
    """

    # Önce ikinci zamiri ara.
    pronouns = []

    for token in tokens:

        info = LEXICON.get(token)

        if info and info["role"] == "PRONOUN":
            pronouns.append(token)

    for pronoun in pronouns:

        if pronoun != subject:
            return pronoun

    # Sonra isim ara.
    nouns = find_nouns(
        tokens,
        subject=subject,
        verb=verb
    )

    if nouns:
        return nouns[0]

    return None


# --------------------------------------------------
# SENTENCE TYPE
# --------------------------------------------------

def detect_sentence_type(tokens):
    """
    Cümle türünü belirler.
    """

    if not tokens:
        return "EMPTY"

    if "MERHABA" in tokens:
        return "GREETING"

    if "TESEKKUR" in tokens:
        return "GREETING"

    if "EVET" in tokens or "HAYIR" in tokens:
        return "ANSWER"

    if "NE" in tokens or "NEREDE" in tokens:
        return "QUESTION"

    if "TEKRAR" in tokens:
        return "CONTROL"

    return "STATEMENT"


# --------------------------------------------------
# VERB CONJUGATION
# --------------------------------------------------

def conjugate(lemma, person):

    conjugations = {

        "sevmek": {
            1: "seviyorum",
            2: "seviyorsun"
        },

        "gelmek": {
            1: "geliyorum",
            2: "geliyorsun"
        },

        "gitmek": {
            1: "gidiyorum",
            2: "gidiyorsun"
        },

        "yemek": {
            1: "yiyorum",
            2: "yiyorsun"
        }
    }

    if lemma not in conjugations:
        return lemma

    return conjugations[lemma].get(person, lemma)


# --------------------------------------------------
# PRONOUN FORMS
# --------------------------------------------------

def pronoun_form(word, position):

    if position == "subject":

        forms = {
            "BEN": "ben",
            "SEN": "sen"
        }

    else:

        forms = {
            "BEN": "beni",
            "SEN": "seni"
        }

    return forms.get(word, word.lower())


# --------------------------------------------------
# MAIN PARSER
# --------------------------------------------------

def parse(words):
    """
    Ana NLP fonksiyonu.

    Örnek:

    ["BEN", "SEN", "SEVMEK"]

    →

    {
        "tokens": [...],
        "roles": {
            "subject": "BEN",
            "object": "SEN",
            "verb": "SEVMEK",
            "time": None,
            "question": None,
            "adjectives": []
        },
        "sentence_type": "STATEMENT"
    }
    """

    tokens = tokenize(words)

    if not tokens:
        return {
            "tokens": [],
            "roles": {
                "subject": None,
                "object": None,
                "verb": None,
                "time": None,
                "question": None,
                "adjectives": []
            },
            "sentence_type": "EMPTY"
        }

    sentence_type = detect_sentence_type(tokens)

    subject = find_subject(tokens)

    verb = find_verb(tokens)

    obj = find_object(
        tokens,
        subject=subject,
        verb=verb
    )

    time = find_time(tokens)

    question = find_question(tokens)

    adjectives = find_adjectives(tokens)

    return {
        "tokens": tokens,

        "roles": {
            "subject": subject,
            "object": obj,
            "verb": verb,
            "time": time,
            "question": question,
            "adjectives": adjectives
        },

        "sentence_type": sentence_type
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    tests = [

        ["BEN", "SEN", "SEVMEK"],

        ["SEN", "BEN", "SEVMEK"],

        ["BEN", "BUGUN", "GELMEK"],

        ["BUGUN", "BEN", "GELMEK"],

        ["SEN", "NEREDE"],

        ["BEN", "SU"],

        ["BEN", "YEMEK"],

        ["BEN", "ARKADAS"],

        ["BEN", "IYI"],

        ["BEN", "KOTU"],

        ["MERHABA"]
    ]

    for words in tests:

        print("\nGirdi:")
        print(words)

        print("\nNLP:")

        result = parse(words)

        print(result)

        print("-" * 50)
