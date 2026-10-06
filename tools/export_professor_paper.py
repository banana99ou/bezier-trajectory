#!/usr/bin/env python3
"""Rebuild the professor-facing MD/PDF and figures from preserved results.

Run from the repo root with .venv/bin/python tools/export_professor_paper.py.
Requires Pandoc, XeLaTeX, and the fonts named in the export header. No optimizer
is run: Figure 4 uses the preserved 400-point trajectories and September 16 rows.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / 'papers/exports/feedback_applied_11c772e'
STEM = ROOT / 'papers/paper_draft_feedback_applied_11c772e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path(tempfile.gettempdir())/'bezier-professor-pdf')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(work/'matplotlib'))
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from tools import build_representative_trajectories as f4

    subprocess.run([sys.executable, str(ROOT/'figures/scp_pipeline.py'), '--save'], cwd=ROOT, check=True)
    curves = json.loads((EXPORT/'data/trajectories.json').read_text())
    raw = [json.loads(line) for line in (ROOT/'doc/results/degree_sensitivity_20260916/raw.jsonl').read_text().splitlines()]
    rows = {x['summary']['scenario']: x['summary'] for x in raw
            if x['summary']['degree'] == f4.T2_DEGREE and x['summary']['n_seg'] == f4.T2_NSEG}

    def decode(a):
        return np.asarray(a) if isinstance(a,list) else np.frombuffer(base64.b64decode(a['bdata']),dtype=a['dtype'])

    solved = []
    for (name,_),color in zip(f4.T2_SCENARIOS, f4.COLORS):
        sc = f4.H.make_scenario(name, N=f4.T2_DEGREE)
        pts = np.column_stack([decode(curves[name][key]) for key in ('x','y','z')])
        assert pts.shape == (400,3)
        k = int(np.argmin(np.linalg.norm(pts,axis=1)))
        margin = float(rows[name]['margin_km'])
        assert abs(float(np.linalg.norm(pts[k]))-sc['r_e']-margin) < 0.002
        info = {'iterations':rows[name]['iters'], 'mean_control_accel_ms2':rows[name]['ctrl_cost_ms2']}
        solved.append((f4.LABEL[name],color,sc,pts,k,margin,0<k<len(pts)-1,info))
    f4.main(solved=solved)
    for name in ('scp_pipeline.png','representative_trajectories.png'):
        shutil.copy2(ROOT/'figures'/name, EXPORT/'figures'/name)

    source = (ROOT/'papers/paper_draft.md').read_text()
    assert source.count('](../figures/') == 5
    manuscript = source.replace('](../figures/', '](exports/feedback_applied_11c772e/figures/')
    STEM.with_suffix('.md').write_text(manuscript)
    for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', manuscript):
        assert (STEM.parent/path).is_file(), path

    tex = work/'paper.tex'
    subprocess.run(['pandoc',str(STEM.with_suffix('.md')),'--standalone','--from=markdown-implicit_figures',
                    '--to=latex','--lua-filter='+str(EXPORT/'export.lua'),
                    '--include-in-header='+str(EXPORT/'header.tex'),
                    '-V','mainfont=Helvetica Neue','-V','mathfont=latinmodern-math.otf',
                    '-V','geometry:margin=0.85in','-V','fontsize=10pt','-V','linestretch=1.1',
                    '-V','colorlinks=true','-o',str(tex)],cwd=STEM.parent,check=True)
    for pass_no in (1,2):
        with (work/f'build-{pass_no}.stdout').open('w') as log:
            subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',
                            '-output-directory='+str(work), str(tex)],cwd=STEM.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(work/'paper.pdf', STEM.with_suffix('.pdf'))
    checksums = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [ROOT/'papers/paper_draft.md',STEM.with_suffix('.md'),STEM.with_suffix('.pdf'),
                              EXPORT/'export.lua',EXPORT/'header.tex',Path(__file__).resolve(),
                              ROOT/'figures/scp_pipeline.py',ROOT/'tools/build_representative_trajectories.py',
                              EXPORT/'data/trajectories.json',*sorted((EXPORT/'figures').glob('*.png'))]}
    (EXPORT/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print(f'Updated {STEM.with_suffix(".md")} and {STEM.with_suffix(".pdf")}')
    print(f'Build log and TeX: {work}')


if __name__ == '__main__':
    main()
