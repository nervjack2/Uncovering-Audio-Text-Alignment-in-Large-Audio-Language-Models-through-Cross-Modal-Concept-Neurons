import re

from tqdm import tqdm

from concept_neurons.prompts import MC_INSTRUCTION

_OPTION = re.compile(r"\(([a-d])\)\s*([A-Za-z]+)")
_ANY_OPTION = re.compile(r"\([a-d]\)")


def mc_prompt(question: str) -> str:
    return f"{MC_INSTRUCTION}\n\n{question}"


def parse_answer(response: str) -> str:
    """The predicted option letter is the first character of the response. A leading "(x)" is
    accepted when it is the only option-like string, i.e. the model is not repeating the options."""
    text = response.strip().lower()
    if not text:
        return ""
    if text[0] == "(" and len(text) > 1 and text[1] in "abcd" and len(_ANY_OPTION.findall(text)) == 1:
        return text[1]
    return text[0]


def option_letter(question: str, label: str) -> str | None:
    for letter, word in _OPTION.findall(question):
        if word.lower() == label.lower():
            return letter
    return None


def predict(model, sample: dict, max_new_tokens: int = 32) -> str:
    inputs = model.prepare_inputs(sample["audio"], mc_prompt(sample["mc_question"]))
    return parse_answer(model.generate(inputs, max_new_tokens=max_new_tokens))


def accuracy(model, dataset, indices: list[int], desc: str = "") -> float:
    correct = [predict(model, dataset[i]) == dataset.info[i]["mc_answer"]
               for i in tqdm(indices, desc=desc, leave=False)]
    return sum(correct) / len(correct)


def target_rate(model, samples: list[dict], label: str) -> float:
    """Fraction of samples answered with the option of `label`."""
    hits = [predict(model, s) == option_letter(s["mc_question"], label) for s in samples]
    return sum(hits) / len(hits)
