"""Plot frozen selected Task cases from exported CSV only; no mask computation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def plot(data, output):
    data, output = Path(data), Path(output)
    source = data/'matched_selected.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['view']=='Task']
    rows.sort(key=lambda r: int(r['selection_rank']))
    if len(rows) > 3 or any(r['selected']!='True' or r['included']!='True' for r in rows):
        raise ValueError('Invalid frozen Task selection')
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(input_file=source.name, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      view='Task', selection='frozen per-view top three, exact equality first; descriptive search',
                      source_data='matched_selected.csv; full search in matched_candidates.csv',
                      suite='16 joint conditions per family pair; not 400-condition JSS',
                      selected_cases=len(rows), inference='none; no selected-case significance intervals')
    if not rows:
        provenance['status']='omitted: no frozen Task cases'
    else:
        # Optional dependency is imported only on the plotting path.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        matplotlib.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
            'axes.labelsize':8, 'axes.titlesize':9, 'xtick.labelsize':7, 'ytick.labelsize':7,
            'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
            'axes.spines.top':False, 'axes.spines.right':False})
        # Wide label gutter protects complete family names at final 180 mm width.
        fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 97/25.4), sharey=True,
                                gridspec_kw={'width_ratios':[1,1]})
        fig.subplots_adjust(left=.305, right=.985, bottom=.21, top=.76, wspace=.2)
        names = {'brightness':'Brightness', 'contrast':'Contrast', 'gaussian_noise':'Gaussian noise',
                 'keyboard':'Keyboard', 'ocr':'OCR', 'whitespace_punc':'Whitespace / punctuation'}
        model_names = {'blip2':'BLIP-2', 'llava':'LLaVA-1.5', 'qwen25':'Qwen2.5-VL', 'qwen3vl':'Qwen3-VL'}
        colors = ('#969DA5','#376C9B')
        labels, positions = [], []
        for case, row in enumerate(rows):
            for j, suffix in enumerate(('a','b')):
                y = (len(rows)-1-case)*2.8 + (1-j)*.84
                positions.append(y)
                image, text = row['pair_'+suffix].split('__')
                labels.append(f"{names.get(image,image)} ×\n{names.get(text,text)}")
                for ax, field, numerator, denominator in (
                        (axes[0], 'mean_accuracy_', 'correct_', 'denominator'),
                        (axes[1], 'pair_suite_survival_', 'survivors_', 'strict_roots')):
                    value = float(row[field+suffix])
                    ax.barh(y, value, height=.64, color=colors[j], zorder=3)
                    ax.text(.018, y, f"{int(row[numerator+suffix])}/{int(row[denominator])}  ({100*value:.2f}%)",
                            va='center', ha='left', color='white', fontsize=7.2, fontweight='medium', zorder=4)
            center=(len(rows)-1-case)*2.8+.42
            axes[0].text(-.035, center+1.04, f"Case {int(row['selection_rank'])} · {model_names.get(row['model'], row['model'])}", ha='right',
                         va='center', transform=axes[0].get_yaxis_transform(), fontsize=7.5, fontweight='bold')
        for ax, title in zip(axes, ('a  Mean accuracy', 'b  Pair-suite survival')):
            ax.set_title(title, loc='left', pad=10, fontweight='bold')
            ax.set_xlim(0,1.01)
            ax.set_xticks([0,.5,1])
            ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.grid(axis='x', color='#E5E7EB', linewidth=.6, zorder=0)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.set_ylim(-.65,(len(rows)-1)*2.8+1.68)
        axes[0].set_yticks(positions, labels)
        axes[1].tick_params(labelleft=False)
        strict = sorted({int(r['strict_roots']) for r in rows})
        fig.text(.305,.945, 'Equal mean accuracy, different whole-pair survival', fontsize=10, fontweight='bold')
        fig.text(.305,.89, f"Task · common4 strict41 · {strict[0]} shared roots · 16 conditions / pair", fontsize=8)
        fig.text(.02,.105,'Frozen top 3 from 1,200 Task candidates (all models / pairs); descriptive selection.',fontsize=7)
        fig.text(.02,.06,'Counts: correct root–condition records (a); roots passing all 16 conditions (b).',fontsize=7)
        for ext in ('svg','pdf','png'):
            fig.savefig(output/f'matched_pair_survival.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
        provenance.update(status='completed',backend='Python/matplotlib',matplotlib=matplotlib.__version__,
                          width_mm=180,height_mm=97,png_dpi=300,svg_text='editable',pdf_fonttype=42,
                          files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
    with (output/'figure_provenance.json').open('x',encoding='utf-8') as stream:
        json.dump(provenance,stream,indent=2)
        stream.write('\n')
    return provenance


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plot(args.data,args.output)
