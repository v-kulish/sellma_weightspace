#!/usr/bin/env bash
# Fetch pinned sources into data/ and models/ (checksums verified by scripts/record_sources.py / manifests/sources.json).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/scala data/belebele data/euroeval_prompts models
[ -d data/swe_blimp ] || { git clone https://github.com/itsup/swe_blimp data/swe_blimp; git -C data/swe_blimp checkout 4770e273fb781fc8a15ca866e05330b43a63ad2c; }
HF=https://huggingface.co
curl -L -o data/idioms.parquet $HF/datasets/liu-nlp/swedish-idioms/resolve/3d33f55db6c45a82a6a237f3cbee6490af08327d/data/train-00000-of-00001.parquet
for l in sv da nb en; do for s in train test; do
  curl -L -o data/scala/${l}_${s}.parquet "$HF/datasets/alexandrainst/scala/resolve/890966a73ecdad5dc3d59777a162e5843e2d9e02/${l}/${s}-00000-of-00001.parquet" || echo "check ScaLA file layout for $l/$s"
done; done
for l in swe_Latn dan_Latn nob_Latn fin_Latn isl_Latn eng_Latn zho_Hans; do
  curl -L -o data/belebele/$l.jsonl $HF/datasets/facebook/belebele/resolve/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/$l.jsonl
done
curl -L -o data/euroeval_prompts/linguistic_acceptability.py https://raw.githubusercontent.com/EuroEval/EuroEval/efb93af3a364dd58e166fb1cb45aa44fb3234297/src/euroeval/prompt_templates/linguistic_acceptability.py
echo "Model: huggingface-cli download google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --local-dir models/gemma-4-12B-it"
echo "Then compare sha256 of every file with manifests/sources.json (python scripts/record_sources.py rewrites it; diff before trusting)."
