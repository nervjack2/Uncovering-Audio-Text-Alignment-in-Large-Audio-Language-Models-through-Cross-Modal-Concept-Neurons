# Cross-Modal Concept Neurons in Large Audio-Language Models

Code for the paper **Uncovering Audio-Text Alignment in Large Audio-Language Models through Cross-Modal Concept Neurons**.

Tzu-Quan Lin, Chih-Kai Yang, Yi-Cheng Lin, Hao Tang, Hung-yi Lee

[[Paper]](https://arxiv.org/abs/XXXX.XXXXX)

The language-model backbone of a large audio-language model (LALM) is pretrained on text alone, yet the
model recognizes acoustic attributes such as emotion, speaker gender or spoken language. We look for FFN
neurons that respond to an attribute both when the model *hears* it and when it *reads* a description of it.

For a label *c* (e.g. *angry*), we average the FFN activations over audio clips of *c* and over text
descriptions of *c* ("The speaker feels angry."), subtract the average of the other labels, and take the
top-*k* neurons of each modality. Their intersection is the set of **cross-modal concept neurons** of *c*.
We then test whether

- the audio and text top-*k* sets overlap more for matching labels than for mismatched ones,
- shifting these neurons along a direction computed **from text alone** changes what the model recognizes from **audio** (steering), and
- recognition breaks down when they are set to zero (deactivation).

Models: Qwen2.5-Omni-7B, DeSTA2.5-Audio, Audio Flamingo 3.
Data: the emotion, animal, gender and language tasks of [SAKURA](https://github.com/ckyang1124/SAKURA), downloaded from the Hugging Face Hub on first use.

## Setup

```bash
pip install -r requirements.txt
```

- **Qwen2.5-Omni** runs with `transformers==4.57.*`.
- **DeSTA2.5-Audio** additionally needs the [DeSTA2.5-Audio](https://github.com/kehanlu/DeSTA2.5-Audio) package (`pip install -e DeSTA2.5-Audio`).
- **Audio Flamingo 3** needs `transformers>=5.0`, so we use a separate environment for it.

One GPU with 24 GB is enough for Qwen2.5-Omni and DeSTA2.5-Audio. Audio Flamingo 3 needs 48 GB.

## Usage

Run every command from this directory, once per model (`qwen2.5-omni-7b`, `desta2.5-audio`, `audio-flamingo-3`).

**1. Extract activations** for the audio clips and the text descriptions of all four tasks:

```bash
python extract_activations.py --model qwen2.5-omni-7b
```

**2. Audio-text overlap** (Tables 1 and 2, Figs. 5-7). This step runs on CPU.

```bash
python overlap.py --model qwen2.5-omni-7b
```

**3. Steering** (Table 3, Figs. 3, 8, 9). There is one run per label, and *k* defaults to the value used in the paper (`configs/steering_k.json`):

```bash
python steer.py --model qwen2.5-omni-7b --task emotion --label happy
```

**4. Deactivation** (Table 4, Figs. 4, 10-12). Each run sweeps *k* from 25 to 256,000:

```bash
python deactivate.py --model qwen2.5-omni-7b --task emotion --audio_only
```

**5. Summarize** steering and deactivation: this prints the significance tests and decline onsets and writes the figures to `results/figures/`.

```bash
python summarize.py
```

`run_all.sh` runs all of the above for one model.

## Code

```
concept_neurons/
  data.py         SAKURA tasks and the validation/test split (20/80, seed 0)
  prompts.py      questions, text descriptions, multiple-choice instruction
  models/         wrappers for the three LALMs
  hooks.py        reading, zeroing and shifting FFN neurons (input of down_proj)
  neurons.py      distinctiveness, top-k, cross-modal concept neurons, random controls
  evaluation.py   multiple-choice evaluation and answer parsing
  stats.py        significance tests
  plots.py        figures
```

Neuron sets are always selected on the validation split. Overlap, steering and deactivation are measured on the test split, and steering and deactivation also use the other tasks for their controls.

## Citation

```bibtex
@article{lin2026crossmodal,
  title   = {Uncovering Audio-Text Alignment in Large Audio-Language Models through Cross-Modal Concept Neurons},
  author  = {Lin, Tzu-Quan and Yang, Chih-Kai and Lin, Yi-Cheng and Tang, Hao and Lee, Hung-yi},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
