#!/usr/bin/env bash
# Usage: ./run_all.sh <model>    (qwen2.5-omni-7b, desta2.5-audio or audio-flamingo-3)
set -euo pipefail
model="$1"

python extract_activations.py --model "$model"
python overlap.py --model "$model"

declare -A labels=(
    [emotion]="disgust sad angry fear happy"
    [animal]="cat cow crow dog frog hen pig rooster sheep"
    [gender]="female male"
    [language]="Chinese English French German Italian Japanese Korean Spanish"
)
for task in emotion animal gender language; do
    for label in ${labels[$task]}; do
        python steer.py --model "$model" --task "$task" --label "$label"
    done
    python deactivate.py --model "$model" --task "$task" --audio_only
done

python summarize.py
