"""Record pinned source revisions, licenses and file checksums (data were fetched at these revisions)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common import write, file_digest, ROOT, git_revision

sources = {
 'model': {'id': 'google/gemma-4-12B-it', 'revision': '43c9d014b231474fa7212dbb48a762566c2b255d', 'license': 'apache-2.0',
           'file': 'models/gemma-4-12B-it/model.safetensors',
           'sha256_expected': '5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d'},
 'swe_blimp': {'url': 'https://github.com/itsup/swe_blimp', 'revision': git_revision(ROOT/'data/swe_blimp'),
               'file': 'data/swe_blimp/swe_blimp_v1.csv', 'license': 'none stated in repository; text not redistributed'},
 'swedish_idioms': {'id': 'liu-nlp/swedish-idioms', 'revision': '3d33f55db6c45a82a6a237f3cbee6490af08327d',
                    'file': 'data/idioms.parquet', 'license': 'apache-2.0'},
 'scala': {'id': 'alexandrainst/scala', 'revision': '890966a73ecdad5dc3d59777a162e5843e2d9e02', 'license': 'cc-by-sa-4.0',
           'files': sorted(str(p.relative_to(ROOT)) for p in (ROOT/'data/scala').glob('*.parquet')),
           'prompt_source': {'repo': 'https://github.com/EuroEval/EuroEval', 'revision': 'efb93af3a364dd58e166fb1cb45aa44fb3234297',
                             'file': 'src/euroeval/prompt_templates/linguistic_acceptability.py',
                             'local_copy': 'data/euroeval_prompts/linguistic_acceptability.py'},
           'note': 'EuroEval-hosted ScaLA dataset IDs returned HTTP 401 at fetch time; original public alexandrainst/scala splits used.'},
 'belebele': {'id': 'facebook/belebele', 'revision': '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2', 'license': 'cc-by-sa-4.0',
              'files': sorted(str(p.relative_to(ROOT)) for p in (ROOT/'data/belebele').glob('*.jsonl'))},
}
for s in sources.values():
    for f in ([s['file']] if 'file' in s and not s['file'].endswith('safetensors') else s.get('files', [])):
        s.setdefault('sha256', {})[f] = file_digest(f)
s = sources['scala']['prompt_source']; s['sha256'] = file_digest(s['local_copy'])
write('manifests/sources.json', sources)
print('ok')
