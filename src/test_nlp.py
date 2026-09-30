from sentence_engine import build_sentence


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
    ["MERHABA"],
    ["TESEKKUR"],
    ["EVET"],
    ["HAYIR"],
]


for words in tests:

    print(
        words,
        "=>",
        build_sentence(words)
    )
