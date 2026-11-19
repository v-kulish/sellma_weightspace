"""Render RESEARCH_NOTE.md to RESEARCH_NOTE.pdf (A4) with headless Chrome.

Handles the Markdown subset used in the note: headings, paragraphs, bullet and numbered lists,
pipe tables, **bold**, *italic*, `code` and [links](url). Relative links are resolved against the
public repository (--base-url), so links in the PDF open the files on GitHub.

Usage: python scripts/research_note_pdf.py [--font-pt 10.0]
"""
import argparse
import html
import pathlib
import re
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])', r'<em>\1</em>', t)
    return t


ITEM = re.compile(r'^(\s*)(- |\d+\. )')


def parse_list(lines, i):
    """Parse a (possibly nested, by indentation) list starting at lines[i]; return (html, next index)."""
    base = len(ITEM.match(lines[i]).group(1))
    tag = 'ol' if ITEM.match(lines[i]).group(2)[0].isdigit() else 'ul'
    items = []
    while i < len(lines):
        m = ITEM.match(lines[i])
        if m and len(m.group(1)) == base:
            items.append([inline(lines[i][m.end():]), '']); i += 1
        elif m and len(m.group(1)) > base:
            sub, i = parse_list(lines, i); items[-1][1] += sub
        elif lines[i].startswith(' ' * (base + 2)) and lines[i].strip() and items:
            items[-1][0] += ' ' + inline(lines[i].strip()); i += 1
        elif not lines[i].strip() and i + 1 < len(lines) and lines[i + 1].startswith(' ' * (base + 2)) and items:
            i += 1
        else:
            break
    return f'<{tag}>' + ''.join(f'<li>{t}{sub}</li>' for t, sub in items) + f'</{tag}>', i


def to_html(md):
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1
        elif l.startswith('#'):
            n = len(l) - len(l.lstrip('#'))
            out.append(f'<h{n}>{inline(l[n:].strip())}</h{n}>'); i += 1
        elif l.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
            head, body = rows[0], [r for r in rows[1:] if not set(''.join(r)) <= set('-: ')]
            out.append('<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in head) + '</tr></thead><tbody>' +
                       ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body) + '</tbody></table>')
        elif re.match(r'\s*(- |\d+\. )', l):
            html_list, i = parse_list(lines, i)
            out.append(html_list)
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r'(#|\||\s*- |\s*\d+\. )', lines[i]):
                para.append(lines[i].strip()); i += 1
            out.append(f'<p>{inline(" ".join(para))}</p>')
    return '\n'.join(out)


CSS = """
@page { size: A4; margin: 13mm 14mm 13mm 14mm; }
body { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: %(pt)spt; line-height: 1.28; color: #111; }
h1 { font-size: 1.45em; margin: 0 0 3pt; }
h2 { font-size: 1.12em; margin: 7pt 0 2pt; border-bottom: 0.5pt solid #bbb; padding-bottom: 1pt; }
p { margin: 0 0 4pt; text-align: justify; }
ul, ol { margin: 0 0 4pt; padding-left: 15pt; }
li { margin: 0 0 1.5pt; text-align: justify; }
code { font-family: 'Liberation Mono', monospace; font-size: 0.86em; }
a { color: #1f4e9e; text-decoration: none; }
table { border-collapse: collapse; width: 100%%; margin: 3pt 0 5pt; font-size: 0.9em; }
th, td { border-bottom: 0.4pt solid #ccc; padding: 1.6pt 4pt; text-align: left; vertical-align: top; }
thead th { border-bottom: 0.8pt solid #333; }
table, li { break-inside: avoid; }
h2 { break-after: avoid; }
li ul, li ol { margin: 1pt 0 0; }
tbody tr:last-child td { border-bottom: 0.8pt solid #333; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--font-pt', type=float, default=10.0)
    ap.add_argument('--src', default='RESEARCH_NOTE.md')
    ap.add_argument('--dst', default='RESEARCH_NOTE.pdf')
    ap.add_argument('--base-url', default='https://github.com/v-kulish/sellma_weightspace/blob/main/')
    a = ap.parse_args()
    body = to_html((ROOT / a.src).read_text())
    body = re.sub(r'href="(?!https?:|#|mailto:)([^"]+)"', lambda m: f'href="{a.base_url}{m.group(1)}"', body)
    doc = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS % {"pt": a.font_pt}}</style></head><body>{body}</body></html>'
    chrome = shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chromium-browser')
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / 'note.html'
        src.write_text(doc)
        subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-sandbox', '--no-pdf-header-footer',
                        f'--user-data-dir={d}/profile', f'--print-to-pdf={ROOT / a.dst}', src.as_uri()],
                       check=True, capture_output=True)
    print(ROOT / a.dst)


if __name__ == '__main__':
    main()
