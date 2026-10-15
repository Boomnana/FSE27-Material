import json
import platform
from pathlib import Path

import build_tables
import common
import plot_case
import plot_rq1
import plot_rq2
import plot_rq3
import plot_rq4


def main():
    root = Path(__file__).resolve().parents[1]
    common.verify_inputs()
    report = build_tables.build()
    for i in range(1, 5):
        name = f'rq{i}.tex'
        assert (root / 'generated/results/tables' / name).read_bytes() == (root / 'paper_artifacts/tables' / name).read_bytes(), name
    for module in (plot_rq1, plot_rq2, plot_rq3, plot_rq4, plot_case):
        module.main()
    report['generated_figures'] = [{
        'file': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size
    } for p in sorted((root / 'generated/figures').glob('*.pdf'))]
    report['python_version'] = platform.python_version()
    report['scope'] = 'Tables 3-6 and Figures 3-7 regenerated from the bundled derived data; Tables 1-2 and Figures 1-2 are supplied as static assets.'
    (root / 'generated/paper_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: four table bodies match the supplied tables; five result figures generated.')
    return report


if __name__ == '__main__':
    main()
