#!/usr/bin/env python3
"""Build a Chinese DOCX from the reviewed Markdown; no Python packages needed.

LibreOffice exports this editable DOCX to PDF. Explicit page breaks in the
source keep each planning topic together. The script never launches experiments.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
ET.register_namespace('w', W)
ET.register_namespace('r', R)
WIDTH = 9638


def el(name, parent=None, text=None, **attrs):
    item = ET.Element('{%s}%s' % (W, name), {'{%s}%s' % (W, k): str(v) for k, v in attrs.items()})
    if text is not None:
        item.text = text
    if parent is not None:
        parent.append(item)
    return item


def xml(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


class Document:
    def __init__(self):
        self.root = el('document')
        self.body = el('body', self.root)
        self.rels = ET.Element('Relationships', xmlns=PKG)
        self.add_rel('styles', 'styles.xml', 'styles')
        self.add_rel('header', 'header1.xml', 'header')
        self.add_rel('footer', 'footer1.xml', 'footer')
        self.links = 0

    def add_rel(self, identifier, target, kind, external=False):
        values = dict(Id=identifier, Target=target, Type=R + '/' + kind)
        if external:
            values['TargetMode'] = 'External'
        ET.SubElement(self.rels, 'Relationship', values)

    def run(self, parent, text, bold=False, code=False, color=None, size=None):
        run = el('r', parent)
        prop = el('rPr', run)
        el('rFonts', prop, ascii='DejaVu Sans Mono' if code else 'Liberation Sans',
           hAnsi='DejaVu Sans Mono' if code else 'Liberation Sans', eastAsia='WenQuanYi Micro Hei')
        el('lang', prop, val='en-US', eastAsia='zh-CN')
        if bold:
            el('b', prop)
        if code:
            el('sz', prop, val=size or 17)
            el('shd', prop, fill='EEF3F7', val='clear')
        elif size:
            el('sz', prop, val=size)
            el('szCs', prop, val=size)
        if color:
            el('color', prop, val=color)
        value = el('t', run, text=text)
        value.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

    def inline(self, paragraph, text, bold=False, color=None, size=None):
        pattern = r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^\)]+\))'
        for token in re.split(pattern, text):
            if not token:
                continue
            if token.startswith('**') and token.endswith('**'):
                self.run(paragraph, token[2:-2], bold=True, color=color, size=size)
            elif token.startswith('`') and token.endswith('`'):
                self.run(paragraph, token[1:-1], bold=bold, code=True, color=color, size=size)
            elif token.startswith('[') and '](' in token:
                match = re.fullmatch(r'\[([^\]]+)\]\(([^\)]+)\)', token)
                self.links += 1
                identifier = 'link' + str(self.links)
                self.add_rel(identifier, match[2], 'hyperlink', True)
                link = el('hyperlink', paragraph)
                link.set('{%s}id' % R, identifier)
                self.run(link, match[1], bold=bold, color=color or '17798A', size=size)
            else:
                self.run(paragraph, token, bold=bold, color=color, size=size)

    def paragraph(self, text, style='Normal', parent=None, **kwargs):
        p = el('p', parent if parent is not None else self.body)
        pp = el('pPr', p)
        el('pStyle', pp, val=style)
        if style in ('Heading1', 'Heading2', 'Heading3', 'Title', 'Subtitle'):
            el('keepNext', pp)
        if style == 'Bullet':
            el('ind', pp, left=220, hanging=180)
            text = '•  ' + text
        self.inline(p, text, **kwargs)
        return p

    def break_page(self):
        p = el('p', self.body)
        pp = el('pPr', p)
        el('spacing', pp, before=0, after=0, line=20, lineRule='exact')
        el('br', el('r', p), type='page')

    def table(self, rows, ratios=None):
        columns = len(rows[0])
        ratios = ratios or ([0.38] + [0.62 / (columns - 1)] * (columns - 1) if columns > 1 else [1])
        if len(ratios) != columns:
            raise ValueError('Table widths do not match columns')
        sizes = [int(WIDTH * r / sum(ratios)) for r in ratios]
        sizes[-1] = WIDTH - sum(sizes[:-1])
        table = el('tbl', self.body)
        tp = el('tblPr', table)
        el('tblW', tp, w=WIDTH, type='dxa')
        el('tblLayout', tp, type='fixed')
        margins = el('tblCellMar', tp)
        for side in ('top', 'bottom'):
            el(side, margins, w=65, type='dxa')
        for side in ('left', 'right'):
            el(side, margins, w=105, type='dxa')
        borders = el('tblBorders', tp)
        for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            el(side, borders, val='single', sz=4, color='D5DFE8')
        grid = el('tblGrid', table)
        for size in sizes:
            el('gridCol', grid, w=size)
        for index, values in enumerate(rows):
            if len(values) != columns:
                raise ValueError('Irregular table row: ' + str(values))
            tr = el('tr', table)
            trp = el('trPr', tr)
            el('cantSplit', trp)
            if not index:
                el('tblHeader', trp)
            for col, value in enumerate(values):
                cell = el('tc', tr)
                cp = el('tcPr', cell)
                el('tcW', cp, w=sizes[col], type='dxa')
                el('vAlign', cp, val='center')
                el('shd', cp, fill='193B52' if not index else ('F0F5F8' if index % 2 else 'FFFFFF'), val='clear')
                self.paragraph(value, 'TableText', parent=cell,
                               bold=not index, color='FFFFFF' if not index else '20313F', size=18)
        self.paragraph('', 'TableAfter')

    def parse(self, markdown):
        lines = markdown.splitlines()
        pos = 0
        ratios = None
        code = False
        while pos < len(lines):
            line = lines[pos].strip()
            pos += 1
            if not line:
                continue
            if line == '<!-- pagebreak -->':
                self.break_page()
                continue
            if line.startswith('<!-- widths:'):
                ratios = [float(v) for v in line.removeprefix('<!-- widths:').removesuffix('-->').strip().split(',')]
                continue
            if line.startswith('```'):
                code = not code
                continue
            if code:
                self.paragraph('`' + line + '`', 'Code')
                continue
            if line.startswith('|'):
                rows = []
                while True:
                    cells = [v.strip() for v in line.strip('|').split('|')]
                    if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                        rows.append(cells)
                    if pos >= len(lines) or not lines[pos].strip().startswith('|'):
                        break
                    line = lines[pos].strip()
                    pos += 1
                self.table(rows, ratios)
                ratios = None
                continue
            if line.startswith('# '):
                self.paragraph(line[2:], 'Title')
            elif line.startswith('## '):
                self.paragraph(line[3:], 'Heading1')
            elif line.startswith('### '):
                self.paragraph(line[4:], 'Heading2')
            elif line.startswith('#### '):
                self.paragraph(line[5:], 'Heading3')
            elif line.startswith('> '):
                self.paragraph(line[2:], 'Callout')
            elif line.startswith('- '):
                self.paragraph(line[2:], 'Bullet')
            else:
                self.paragraph(line)
        section = el('sectPr', self.body)
        for kind in ('header', 'footer'):
            reference = el(kind + 'Reference', section, type='default')
            reference.set('{%s}id' % R, kind)
        el('pgSz', section, w=11906, h=16838)
        el('pgMar', section, top=1134, right=1134, bottom=1134, left=1134, header=440, footer=500)

    def styles(self):
        root = el('styles')
        defaults = el('docDefaults', root)
        rp = el('rPr', el('rPrDefault', defaults))
        el('rFonts', rp, ascii='Liberation Sans', hAnsi='Liberation Sans', eastAsia='WenQuanYi Micro Hei')
        el('sz', rp, val=21)
        el('lang', rp, val='en-US', eastAsia='zh-CN')
        variants = {
            'Normal': (21, '20313F', 0, 90, 300, False),
            'Title': (46, '193B52', 440, 360, 650, True),
            'Subtitle': (26, '17798A', 0, 260, 420, False),
            'Heading1': (32, '193B52', 0, 240, 440, True),
            'Heading2': (25, '17798A', 160, 120, 360, True),
            'Heading3': (22, '193B52', 130, 90, 330, True),
            'Bullet': (21, '20313F', 0, 70, 300, False),
            'Callout': (23, '193B52', 130, 180, 360, True),
            'Code': (17, '20313F', 0, 30, 240, False),
            'TableText': (18, '20313F', 0, 20, 250, False),
            'TableAfter': (2, 'FFFFFF', 0, 90, 20, False),
            'Header': (17, '60788A', 0, 0, 220, False),
            'Footer': (17, '60788A', 0, 0, 220, False),
        }
        for identifier, (size, color, before, after, line, bold) in variants.items():
            style = el('style', root, type='paragraph', styleId=identifier)
            if identifier == 'Normal':
                style.set('{%s}default' % W, '1')
            el('name', style, val=identifier)
            if identifier != 'Normal':
                el('basedOn', style, val='Normal')
            pp = el('pPr', style)
            el('spacing', pp, before=before, after=after, line=line, lineRule='exact')
            el('widowControl', pp)
            if identifier.startswith('Heading'):
                el('outlineLvl', pp, val=int(identifier[-1]) - 1)
                el('keepNext', pp)
            if identifier == 'Callout':
                el('shd', pp, fill='E8F2F4', val='clear')
                el('ind', pp, left=180, right=180)
            if identifier == 'Footer':
                el('jc', pp, val='center')
            if identifier == 'Header':
                borders = el('pBdr', pp)
                el('bottom', borders, val='single', sz=6, color='D5DFE8', space=8)
            rp = el('rPr', style)
            el('sz', rp, val=size)
            el('szCs', rp, val=size)
            el('color', rp, val=color)
            if bold:
                el('b', rp)
        return root

    def save(self, destination):
        header = el('hdr')
        self.paragraph('TrustGTR  /  项目完整计划', 'Header', parent=header)
        footer = el('ftr')
        p = self.paragraph('项目计划 v1.0 · 2026-10-05     第 ', 'Footer', parent=footer, size=17, color='60788A')
        el('fldSimple', p, instr='PAGE')
        self.run(p, ' 页 / 共 ', size=17, color='60788A')
        el('fldSimple', p, instr='NUMPAGES')
        self.run(p, ' 页', size=17, color='60788A')
        types = ET.Element('Types', xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
        ET.SubElement(types, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
        ET.SubElement(types, 'Default', Extension='xml', ContentType='application/xml')
        for part, kind in [('document','document.main'),('styles','styles'),('header1','header'),('footer1','footer')]:
            ET.SubElement(types, 'Override', PartName='/word/' + part + '.xml',
                          ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.' + kind + '+xml')
        pkg = ET.Element('Relationships', xmlns=PKG)
        ET.SubElement(pkg, 'Relationship', Id='document', Type=R+'/officeDocument', Target='word/document.xml')
        settings = el('settings')
        el('updateFields', settings, val='true')
        ET.SubElement(types, 'Override', PartName='/word/settings.xml',
                      ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml')
        self.add_rel('settings', 'settings.xml', 'settings')
        with ZipFile(destination, 'w', ZIP_DEFLATED) as z:
            for name, value in {'[Content_Types].xml':types, '_rels/.rels':pkg,
                                'word/document.xml':self.root, 'word/styles.xml':self.styles(),
                                'word/_rels/document.xml.rels':self.rels, 'word/header1.xml':header,
                                'word/footer1.xml':footer, 'word/settings.xml':settings}.items():
                z.writestr(name, xml(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    values, formal, smoke, failures = {}, [], [], []
    records = {}
    for path in sorted((root / 'runs').glob('*/metrics.json')):
        metric = json.loads(path.read_text())
        if metric.get('status') == 'SUCCESS':
            if '__forecast__' in path.parent.name:
                formal.append(metric)
                parts = path.parent.name.split('__')
                model, dataset = parts[0], parts[1]
                label = model.removeprefix('trustgtr_') if model.startswith('trustgtr_') else model
                label = {'a':'a1', 'b':'b1', 'ac':'ac1', 'abc':'abc1'}.get(label, label)
                key = dataset + '_' + label
                records[key] = metric
                values[key + '_id'] = path.parent.name
                for field in ('mse','mae','parameters','peak_gpu_memory_mib','runtime_seconds',
                              'epochs_completed','best_epoch','inference_time_seconds'):
                    value = metric.get(field)
                    if value is None:
                        continue
                    if field in ('mse','mae'):
                        value = f'{value:.6f}'
                    elif field == 'parameters':
                        value = f'{value:,}'
                    elif field in ('runtime_seconds','peak_gpu_memory_mib'):
                        value = f'{value:.1f}'
                    elif field == 'inference_time_seconds':
                        value = f'{value:.5f}'
                    values[key + '_' + field] = str(value)
            else:
                smoke.append(metric)
        elif metric.get('status') == 'CODE_ERROR':
            failures.append(metric)
    for key, metric in records.items():
        dataset = key.split('_')[0]
        baseline = records.get(dataset + '_gtr')
        if baseline is None:
            continue
        for field in ('mse','mae'):
            values[key + '_' + field + '_gain'] = f'{(baseline[field] - metric[field])/baseline[field]*100:.4f}%'
            values[key + '_' + field + '_delta'] = f'{metric[field] - baseline[field]:+.6f}'
    for dataset, paper_mse, paper_mae in [('etth1',0.367,0.391),('solar',0.176,0.235)]:
        baseline = records[dataset + '_gtr']
        values[dataset + '_paper_mse_delta'] = f'{baseline["mse"]-paper_mse:+.6f}'
        values[dataset + '_paper_mae_delta'] = f'{baseline["mae"]-paper_mae:+.6f}'
    values.update(successful_formal_count=str(len(formal)), successful_smoke_count=str(len(smoke)),
                  code_error_records=str(len(failures)))
    queue = json.loads((root/'runtime/queue_state.json').read_text())
    values['queue_status'] = queue['daemon']['status']
    values['queue_success_count'] = str(queue['counts']['succeeded'])
    source_text = args.source.read_text(encoding='utf-8')
    def substitute(match):
        key = match[1].strip()
        if key not in values:
            raise ValueError('Missing verified evidence token: ' + key)
        return values[key]
    source_text = re.sub(r'\{\{([^}]+)\}\}', substitute, source_text)
    doc = Document()
    doc.parse(source_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    args.output.with_suffix('.md').write_text(source_text, encoding='utf-8')
    evidence = dict(as_of='2026-10-05', metrics=records, queue_snapshot=dict(
        config_path=queue['config_path'],counts=queue['counts'],daemon_status=queue['daemon']['status']),
        source_hashes={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (root/'runs').glob('*/metrics.json')})
    args.output.with_suffix('.evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
