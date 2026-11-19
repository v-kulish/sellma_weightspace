# Third-party sources and licenses

The MIT license covers original code only. Weights and benchmark data are not committed; `manifests/sources.json`
pins revisions and SHA-256 checksums. Item records in `results/` contain ids, hashes, scores and (for ScaLA)
model outputs; benchmark text is not redistributed.

| Source | Revision | License / handling |
|---|---|---|
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | `43c9d014b231474fa7212dbb48a762566c2b255d` | Apache 2.0 (model card). Weights stay local. Edit artifacts in `artifacts/` are low-rank factors derived from these weights (Apache 2.0). |
| [Swe-BLiMP](https://github.com/itsup/swe_blimp) (Sjons, Heinat, Kurfalı 2026, LREC) | `4770e273fb781fc8a15ca866e05330b43a63ad2c` | No license stated in the repository: sentences are not redistributed. |
| [liu-nlp/swedish-idioms](https://huggingface.co/datasets/liu-nlp/swedish-idioms) (TACL 2026) | `3d33f55db6c45a82a6a237f3cbee6490af08327d` | Apache 2.0. |
| [alexandrainst/scala](https://huggingface.co/datasets/alexandrainst/scala) (Nielsen 2023) | `890966a73ecdad5dc3d59777a162e5843e2d9e02` | CC BY-SA 4.0. |
| [EuroEval](https://github.com/EuroEval/EuroEval) prompt wording | `efb93af3a364dd58e166fb1cb45aa44fb3234297` | MIT; `linguistic_acceptability.py` prompt templates read locally. |
| [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) (Bandarkar et al. 2024) | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` | CC BY-SA 4.0. |
