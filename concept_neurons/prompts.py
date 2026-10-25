QUESTIONS = {
    "emotion": "What is the emotional tone of the speaker in the audio?",
    "animal": "What animal is making the sound in the audio?",
    "gender": "What is the gender of the speaker in the audio?",
    "language": "What language is the speaker speaking in the audio?",
}

MC_INSTRUCTION = (
    "Listen to the audio and answer the multiple-choice question below. "
    "Reply with exactly one letter: a, b, c, or d. "
    "Output nothing else -- no words, no punctuation."
)


def _emotion(word: str, article: str = "a") -> list[str]:
    return [
        f"The speaker feels {word}.",
        f"The speaker sounds {word} right now.",
        f"This is {article} {word} voice.",
        f"The speaker is expressing {article} {word} tone.",
        f"The overall emotion in the speaker's voice is {word}.",
    ]


def _emotion_noun(word: str) -> list[str]:
    return [
        f"The speaker feels {word}.",
        f"The speaker's voice is full of {word}.",
        f"This voice conveys a sense of {word}.",
        f"The speaker is expressing {word}.",
        f"The overall emotion in the speaker's voice is {word}.",
    ]


ANIMAL_SOUNDS = {"cat": "meowing", "cow": "mooing", "crow": "cawing", "dog": "barking", "frog": "croaking",
                 "hen": "clucking", "pig": "oinking", "rooster": "crowing", "sheep": "bleating"}


def _animal(animal: str) -> list[str]:
    return [
        f"The sound in the audio is from a {animal}.",
        f"This audio contains the sound of a {animal} {ANIMAL_SOUNDS[animal]}.",
        f"A {animal} is making the sound in this audio.",
        f"The animal making this sound is a {animal}.",
        f"This is the sound of a {animal}.",
    ]


def _gender(gender: str) -> list[str]:
    return [
        f"The speaker in the audio is {gender}.",
        f"This is a {gender} voice.",
        f"The voice belongs to a {gender} speaker.",
        f"The speaker is {gender}.",
        f"The gender of the speaker is {gender}.",
    ]


def _language(language: str) -> list[str]:
    return [
        f"The speaker is speaking {language}.",
        f"This audio is in the {language} language.",
        f"The language spoken in the audio is {language}.",
        f"The speaker is using {language}.",
        f"{language} is being spoken in this audio.",
    ]


# Five text descriptions per label (paper, Appendix "Text Descriptions").
DESCRIPTIONS = {
    "emotion": {
        "happy": _emotion("happy"),
        "sad": _emotion("sad"),
        "angry": _emotion("angry", "an"),
        "fear": _emotion_noun("fear"),
        "disgust": _emotion_noun("disgust"),
    },
    "animal": {a: _animal(a) for a in ANIMAL_SOUNDS},
    "gender": {g: _gender(g) for g in ["female", "male"]},
    "language": {l: _language(l) for l in
                 ["Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Spanish"]},
}

# Descriptions of the two synthetic steering carriers.
CARRIER_DESCRIPTIONS = {
    "silence": [
        "There is no audio here.",
        "This audio contains complete silence.",
        "No sound can be heard in this audio.",
        "The audio track is silent.",
        "This is a silent audio with no sound at all.",
    ],
    "noise": [
        "There is a noisy audio.",
        "This audio contains only random noise.",
        "The audio is filled with static noise.",
        "This is a noisy audio with no clear sound.",
        "The sound in this audio is just noise.",
    ],
}
