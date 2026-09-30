# src/sentence_engine.py

from nlp_engine import parse, conjugate, pronoun_form


WORD_MAP = {
    "BEN": "ben",
    "SEN": "sen",
    "SU": "su",
    "YEMEK": "yemek",
    "ARKADAS": "arkadaş",
    "YARDIM": "yardım",
    "BUGUN": "bugün",
    "YARIN": "yarın",
    "IYI": "iyi",
    "KOTU": "kötü",
}


def capitalize_sentence(sentence):
    """
    Cümlenin ilk harfini büyütür.
    """

    if not sentence:
        return ""

    return sentence[0].upper() + sentence[1:]


def build_sentence(words):
    """
    GRU'dan gelen kelimeleri NLP ile analiz eder
    ve Türkçe cümle üretir.
    """

    if not words:
        return ""

    nlp = parse(words)

    tokens = nlp["tokens"]
    roles = nlp["roles"]
    sentence_type = nlp["sentence_type"]

    if not tokens:
        return ""

    subject = roles["subject"]
    obj = roles["object"]
    verb = roles["verb"]
    time = roles["time"]
    question = roles["question"]
    adjectives = roles["adjectives"]

    # ==================================================
    # GREETING
    # ==================================================

    if sentence_type == "GREETING":

        if "MERHABA" in tokens:
            return "Merhaba."

        if "TESEKKUR" in tokens:
            return "Teşekkür ederim."

    # ==================================================
    # ANSWER
    # ==================================================

    if sentence_type == "ANSWER":

        if "EVET" in tokens:
            return "Evet."

        if "HAYIR" in tokens:
            return "Hayır."

    # ==================================================
    # QUESTION
    # ==================================================

    if sentence_type == "QUESTION":

        if question == "NEREDE":

            if subject == "SEN":
                return "Sen neredesin?"

            if subject == "BEN":
                return "Ben neredeyim?"

            return "Nerede?"

        if question == "NE":
            return "Ne?"

    # ==================================================
    # CONTROL
    # ==================================================

    if sentence_type == "CONTROL":

        if "TEKRAR" in tokens:
            return "Tekrar."

    # ==================================================
    # SUBJECT
    # ==================================================

    subject_text = ""

    if subject:
        subject_text = pronoun_form(
            subject,
            "subject"
        )

    # ==================================================
    # OBJECT
    # ==================================================

    object_text = ""

    if obj:

        if obj in ["BEN", "SEN"]:

            object_text = pronoun_form(
                obj,
                "object"
            )

        elif obj in WORD_MAP:

            object_text = WORD_MAP[obj]

    # ==================================================
    # TIME
    # ==================================================

    time_text = ""

    if time == "BUGUN":
        time_text = "bugün"

    elif time == "YARIN":
        time_text = "yarın"

    # ==================================================
    # VERB
    # ==================================================

    verb_text = ""

    if verb:

        lemma_map = {
            "SEVMEK": "sevmek",
            "GELMEK": "gelmek",
            "GITMEK": "gitmek",
            "YEMEK": "yemek",
        }

        lemma = lemma_map.get(verb)

        if lemma:

            person = 1

            if subject == "SEN":
                person = 2

            verb_text = conjugate(
                lemma,
                person
            )

    # ==================================================
    # SEVMEK
    # ==================================================

    if verb == "SEVMEK":

        parts = []

        if subject_text:
            parts.append(subject_text)

        if time_text:
            parts.append(time_text)

        if object_text:
            parts.append(object_text)

        if verb_text:
            parts.append(verb_text)

        if parts:
            return capitalize_sentence(
                " ".join(parts)
            ) + "."

    # ==================================================
    # GELMEK / GITMEK
    # ==================================================

    if verb in ["GELMEK", "GITMEK"]:

        parts = []

        if subject_text:
            parts.append(subject_text)

        if time_text:
            parts.append(time_text)

        if verb_text:
            parts.append(verb_text)

        if parts:
            return capitalize_sentence(
                " ".join(parts)
            ) + "."

    # ==================================================
    # YEMEK
    # ==================================================

    if verb == "YEMEK":

        parts = []

        if subject_text:
            parts.append(subject_text)

        if time_text:
            parts.append(time_text)

        parts.append("yemek")

        if subject == "BEN":
            parts.append("yiyorum")

        elif subject == "SEN":
            parts.append("yiyorsun")

        else:
            parts.append("yiyor")

        return capitalize_sentence(
            " ".join(parts)
        ) + "."

    # ==================================================
    # SU
    # ==================================================

    if "SU" in tokens:

        if subject == "BEN":
            return "Ben su istiyorum."

        if subject == "SEN":
            return "Sen su istiyorsun."

        return "Su."

    # ==================================================
    # YARDIM
    # ==================================================

    if "YARDIM" in tokens:
        return "Yardım."

    # ==================================================
    # ADJECTIVES
    # ==================================================

    if adjectives:

        if "IYI" in adjectives:

            if subject == "BEN":
                return "Ben iyiyim."

            if subject == "SEN":
                return "Sen iyisin."

            return "İyi."

        if "KOTU" in adjectives:

            if subject == "BEN":
                return "Ben kötüyüm."

            if subject == "SEN":
                return "Sen kötüsün."

            return "Kötü."

    # ==================================================
    # ARKADAS
    # ==================================================

    if "ARKADAS" in tokens:

        if subject == "BEN":
            return "Ben arkadaşım."

        if subject == "SEN":
            return "Sen arkadaşımsın."

        return "Arkadaş."

    # ==================================================
    # GENEL FALLBACK
    # ==================================================

    result = []

    for token in tokens:

        if token in WORD_MAP:
            result.append(WORD_MAP[token])

    if result:

        return capitalize_sentence(
            " ".join(result)
        ) + "."

    return capitalize_sentence(
        " ".join(tokens)
    ) + "."
