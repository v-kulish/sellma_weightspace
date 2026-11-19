"""Render reports/research_note.md to PDF with local ReportLab + DejaVu fonts (fonts-dejavu-core).
Supports headings, paragraphs, bullet lists, pipe tables, **bold**, `code`, [links](url)."""
import html
import re
import sys

from .common import ROOT


def main(src='reports/research_note.md', dst='reports/research_note.pdf'):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuMono', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu', boldItalic='DejaVu-Bold')
    st = getSampleStyleSheet()
    body = ParagraphStyle('B', parent=st['BodyText'], fontName='DejaVu', fontSize=8.6, leading=11.4, spaceAfter=2)
    bullet = ParagraphStyle('L', parent=body, leftIndent=10, bulletIndent=2)
    small = ParagraphStyle('S', parent=body, fontSize=7.2, leading=9)
    title = ParagraphStyle('T', parent=body, fontName='DejaVu-Bold', fontSize=13.5, leading=17, spaceAfter=4)
    h2 = ParagraphStyle('H2', parent=body, fontName='DejaVu-Bold', fontSize=9.8, leading=12.5, spaceBefore=6, spaceAfter=2)

    def mk(t):
        t = html.escape(t)
        t = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#225588">\1</link>', t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
        t = re.sub(r'(?<![\w*])\*([^*\n]+?)\*(?![\w*])', r'\1', t)
        return re.sub(r'`([^`]+)`', r'<font name="DejaVuMono">\1</font>', t)

    lines = (ROOT / src).read_text().splitlines()
    story, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            story.append(Spacer(1, 2)); i += 1; continue
        if ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append([Paragraph(mk(c), small) for c in cells])
                i += 1
            n = len(rows[0]); w = 174 * mm
            first = min(0.34, 1.6 / n) * w
            cw = [first] + [(w - first) / (n - 1)] * (n - 1) if n > 1 else [w]
            t = Table(rows, colWidths=cw, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9edf1')),
                                   ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.grey),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5), ('TOPPADDING', (0, 0), (-1, -1), 2.5)]))
            story.append(t); story.append(Spacer(1, 3)); continue
        if ln.startswith('# '):
            story.append(Paragraph(mk(ln[2:]), title))
        elif ln.startswith('## ') or ln.startswith('### '):
            story.append(Paragraph(mk(ln.lstrip('#').strip()), h2))
        elif re.match(r'^\s*([-*]|\d+\.)\s+', ln):
            m = re.match(r'^\s*([-*]|\d+\.)\s+(.*)', ln)
            text = m.group(2)
            while i + 1 < len(lines) and lines[i + 1].startswith('  ') and lines[i + 1].strip() and \
                    not re.match(r'^\s*([-*]|\d+\.)\s+', lines[i + 1]):
                i += 1; text += ' ' + lines[i].strip()
            story.append(Paragraph(mk(text), bullet, bulletText='•' if m.group(1) in '-*' else m.group(1)))
        else:
            p = ln
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(('#', '|')) and \
                    not re.match(r'^\s*([-*]|\d+\.)\s+', lines[i + 1]):
                i += 1; p += ' ' + lines[i]
            story.append(Paragraph(mk(p), body))
        i += 1
    doc = SimpleDocTemplate(str(ROOT / dst), pagesize=(210 * mm, 297 * mm), rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title='Targeted weight editing for Swedish grammatical preferences (Gemma 4 12B IT)')

    def footer(c, d):
        c.setFont('DejaVu', 7); c.drawRightString(192 * mm, 8 * mm, str(d.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(ROOT / dst)


if __name__ == '__main__':
    main(*sys.argv[1:])
