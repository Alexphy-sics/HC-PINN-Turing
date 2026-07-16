"""build_paper.py - Final paper assembly with fig0 embedded.
Run from: C:/Users/ROG/Desktop/HC_PINN_Turing
"""
import os, sys, re, shutil, zipfile, subprocess

PROJECT = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.join(os.path.expanduser('~'), 'Desktop')
SRC_DOCX = os.path.join(DESKTOP, 'HC_PINN_Turing_Paper.docx')
FIG0_PNG = os.path.join(PROJECT, 'results', 'figures', 'fig0_dispersion_relation.png')
TEMP_DIR = os.path.join(os.environ.get('TEMP', 'C:/Windows/Temp'), 'paper_build')

def step1_gen_fig0():
    print('[1/3] Generating dispersion relation figure...')
    r = subprocess.run([sys.executable, 'experiments/fig0_dispersion.py'],
                       cwd=PROJECT, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print('ERROR:', r.stderr)
        sys.exit(1)
    if not os.path.exists(FIG0_PNG):
        print('ERROR: fig0 PNG not generated')
        sys.exit(1)
    print('  OK')

def step2_rebuild_docx():
    print('[2/3] Rebuilding docx with fig0 embedded...')

    # Clean temp
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # Unpack existing docx
    with zipfile.ZipFile(SRC_DOCX, 'r') as z:
        z.extractall(TEMP_DIR)

    # Copy fig0 into media
    media_dir = os.path.join(TEMP_DIR, 'word', 'media')
    os.makedirs(media_dir, exist_ok=True)
    shutil.copy(FIG0_PNG, os.path.join(media_dir, 'fig0.png'))
    print('  fig0.png copied')

    # Add relationship in .rels
    rels_path = os.path.join(TEMP_DIR, 'word', '_rels', 'document.xml.rels')
    with open(rels_path, 'r', encoding='utf-8') as f:
        rels = f.read()

    all_ids = re.findall(r'Id="(rId\d+)"', rels)
    max_id = max(int(r[3:]) for r in all_ids)
    next_rid = 'rId%d' % (max_id + 1)
    print('  Next rId: %s' % next_rid)

    new_rel = '<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig0.png"/>' % next_rid
    rels = rels.replace('</Relationships>', '  %s\n</Relationships>' % new_rel)
    with open(rels_path, 'w', encoding='utf-8') as f:
        f.write(rels)
    print('  Relationship added')

    # Read document.xml
    doc_path = os.path.join(TEMP_DIR, 'word', 'document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        xml = f.read()

    # Get drawing template from first existing image
    ds = xml.find('<w:drawing>')
    de = xml.find('</w:drawing>', ds) + len('</w:drawing>')
    template = xml[ds:de]
    drawing = template.replace('rId1', next_rid)
    drawing = re.sub(r'<wp:docPr id="\d+"', '<wp:docPr id="999"', drawing)

    # Find insertion point: before "4. Results" heading
    insert_at = None
    for marker in ['4. Results', '4.1 FDM Benchmark Solutions']:
        idx = xml.find(marker)
        if idx > 0:
            ps = xml.rfind('<w:p>', 0, idx)
            insert_at = ps
            break

    if insert_at is None:
        print('ERROR: cannot find insertion point')
        sys.exit(1)

    # Build figure XML with caption
    gamma_char = 'γ'
    lam_char = 'λ'
    caption = (
        'Figure 0. Linear dispersion relation for the Schnakenberg model '
        'at %s = 220 and 900. Shaded regions mark the unstable wavenumber '
        'band (Re(%s) > 0). Annotations mark the most unstable wavenumber '
        'k* for each regime. Inset: low-k detail showing the stability '
        'threshold crossing.' % (gamma_char, lam_char)
    )

    fig_block = (
        '<w:p><w:pPr><w:spacing w:before="160" w:after="160"/>'
        '<w:jc w:val="center"/></w:pPr>%s</w:p>'
        '<w:p><w:pPr><w:spacing w:after="160"/><w:jc w:val="center"/></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" '
        'w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
        '<w:i/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
        '<w:t xml:space="preserve">%s</w:t></w:r></w:p>'
    ) % (drawing, caption)

    xml = xml[:insert_at] + fig_block + xml[insert_at:]
    print('  Figure inserted at offset %d' % insert_at)

    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(xml)
    print('  document.xml updated')

    # Repack
    if os.path.exists(SRC_DOCX):
        os.remove(SRC_DOCX)
    with zipfile.ZipFile(SRC_DOCX, 'w', zipfile.ZIP_DEFLATED) as zout:
        for root, dirs, files in os.walk(TEMP_DIR):
            for fn in files:
                fp = os.path.join(root, fn)
                an = os.path.relpath(fp, TEMP_DIR).replace(os.sep, '/')
                zout.write(fp, an)
    print('  Docx saved to Desktop')

def step3_update_final():
    print('[3/3] Updating HC_PINN_final folder...')
    import glob
    final_dir = os.path.join(DESKTOP, 'HC_PINN_final')
    for sd in ['paper', 'figures_pdf', 'figures_png']:
        sd2 = {'paper': '论文', 'figures_pdf': '图表_PDF', 'figures_png': '图表_PNG'}[sd]
        os.makedirs(os.path.join(final_dir, sd2), exist_ok=True)

    shutil.copy(SRC_DOCX, os.path.join(final_dir, '论文', 'HC_PINN_Turing_Paper.docx'))

    for ext, sd in [('.pdf', '图表_PDF'), ('.png', '图表_PNG')]:
        src 