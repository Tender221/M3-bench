"""Python-only figures of the refreshed, source-linked descriptive summaries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

MODELS=('blip2','llava','qwen25','qwen3vl')
NAMES=dict(zip(MODELS,('BLIP-2','LLaVA','Qwen2.5-VL','Qwen3-VL')))
IMAGES=('gaussian_blur','gaussian_noise','jpeg','brightness','contrast')
TEXTS=('typo','keyboard','ocr','whitespace_punc','casing')
IL=('Blur','Noise','JPEG','Brightness','Contrast')
TL=('Typo','Keyboard','OCR-like','Space/punct.','Casing')
INK='#23313a'; BLUE='#446d89'; TEAL='#39796d'; RUST='#b9644b'; GRAY='#99a4ac'
SEQ=LinearSegmentedColormap.from_list('binary_gap',['#fffaf5','#f4d1b8','#d68d68','#8c3b31'])
DIV=LinearSegmentedColormap.from_list('binary_excess',['#315d82','#c5d9e5','#fbfaf7','#ebc2ae','#9c4437'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.linewidth':0.65,'text.color':INK,'axes.labelcolor':INK,'savefig.facecolor':'white'})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    s=pd.read_csv(a.data/'model_summary.csv').set_index('model')
    cells=pd.read_csv(a.data/'cell_metrics.csv'); pairs=pd.read_csv(a.data/'pair_metrics.csv')
    acc=pd.read_csv(a.data/'accuracy_axes.csv'); slices=pd.read_csv(a.data/'slice_coverage.csv')
    outputs=[]
    def save(fig,name,description):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        invalid=[]
        for text in fig.findobj(matplotlib.text.Text):
            if text.get_visible() and text.get_text().strip():
                box=text.get_window_extent(renderer)
                if not np.isfinite(box.extents).all(): invalid.append(text.get_text())
        assert not invalid,invalid
        paths=[]
        for ext in ('png','pdf','svg'):
            p=a.output/(name+'.'+ext); fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.06)
            paths.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        outputs.append(dict(name=name,description=description,exports=paths,finite_text_bounds=True))
        plt.close(fig)
    # Accuracy axes stay explicit: a format failure is not silently described as semantic failure.
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,87/25.4),layout='constrained')
    x=np.arange(4); w=.23
    for offset,category,color in ((-1,'clean',BLUE),(0,'single',GRAY),(1,'joint',TEAL)):
        values=np.array([s.loc[m,category+'_accuracy']*100 for m in MODELS])
        axs[0].bar(x+offset*w,values,w,color=color,label=category.capitalize())
    axs[0].set(xticks=x,xticklabels=[NAMES[m] for m in MODELS],ylim=(0,105),ylabel='Task accuracy (%)',title='a  Overall correctness and format compliance')
    axs[0].tick_params(axis='x',rotation=22); axs[0].legend(ncol=3,fontsize=6,loc='upper left')
    vals=[100*s.loc[m,'affected_fraction'] for m in MODELS]
    axs[1].bar(x,vals,color=RUST,width=.58)
    for i,m in enumerate(MODELS):
        r=s.loc[m]; axs[1].text(i,vals[i]+2,f"{int(r.affected_strict41_roots)}/{int(r.strict41_roots)}",ha='center',fontsize=7)
    axs[1].set(xticks=x,xticklabels=[NAMES[m] for m in MODELS],ylim=(0,100),ylabel='Affected strict41 questions (%)',title='b  Failures anywhere in the joint suite')
    axs[1].tick_params(axis='x',rotation=22)
    save(fig,'01_accuracy_and_suite_coverage','Pooled task accuracy and distinct-question failure coverage on each model-own strict41 cohort; no confidence intervals implied.')

    def panels(name,frames,title,value,lim=None,fmt='.1f',percent=True,notes=''):
        arrays=[]
        for m in MODELS:
            z=frames[frames.model.eq(m)].pivot(index='image_op',columns='text_op',values=value).reindex(index=IMAGES,columns=TEXTS).to_numpy(float)
            arrays.append(z*(100 if percent else 1))
        maximum=lim or max(float(np.nanmax(z)) for z in arrays)
        fig,axs=plt.subplots(2,2,figsize=(183/25.4,165/25.4),layout='constrained')
        for ax,m,z in zip(axs.flat,MODELS,arrays):
            im=ax.imshow(z,cmap=SEQ,vmin=0,vmax=maximum,interpolation='nearest')
            n=int(s.loc[m,'strict41_roots']) if 'strict41' in name else 1500
            ax.set_title(f'{NAMES[m]} · n = {n} questions',loc='left')
            ax.set(xticks=range(5),xticklabels=TL,yticks=range(5),yticklabels=IL)
            ax.tick_params(axis='x',rotation=32,length=0); ax.tick_params(axis='y',length=0)
            for i,j in np.ndindex(z.shape):
                ax.text(j,i,format(z[i,j],fmt) if np.isfinite(z[i,j]) else 'NA',ha='center',va='center',fontsize=7,color='white' if z[i,j]>.57*maximum else INK)
        fig.suptitle(title,fontsize=9)
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.75,label=notes)
        save(fig,name,title+'; '+notes+'; exact counts and eligible denominators in pair_metrics.csv.')
    panels('02_pair_conditional_joint_gaps',pairs[pairs.stratum.eq('all')],'Joint failure despite both corresponding single inputs passing','gap_rate',notes='Gap / corresponding-single eligible (%)')
    z=pairs[pairs.stratum.eq('strict41')].copy(); z['root_coverage']=z.gap_roots/z.total_roots
    panels('03_pair_strict41_question_coverage',z,'Which corruption pairs expose single-qualified questions?','root_coverage',lim=100,notes='Questions failing within a pair / strict41 questions (%)')

    strict=cells[cells.stratum.eq('strict41')]
    severity=[]
    for m in MODELS:
        z=strict[strict.model.eq(m)].groupby(['image_strength','text_strength'])[['gap','total']].sum()
        severity.append((100*z.gap/z.total).unstack().reindex(index=range(1,5),columns=range(1,5)).to_numpy())
    vmax=max(np.max(z) for z in severity)
    fig,axs=plt.subplots(2,2,figsize=(150/25.4,132/25.4),layout='constrained')
    for ax,m,z in zip(axs.flat,MODELS,severity):
        im=ax.imshow(z,cmap=SEQ,vmin=0,vmax=vmax)
        ax.set(xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5),xlabel='Text severity',ylabel='Image severity',title=NAMES[m])
        for i,j in np.ndindex(z.shape): ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',color='white' if z[i,j]>.57*vmax else INK,fontsize=7)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.75,label='Failures / (25 pairs × strict41 questions), %')
    save(fig,'04_strict41_severity_density','Exact-coordinate event density, pooled over all25 families. No severity monotonicity assumed.')

    miss=pd.read_csv(a.data/'missed_locations.csv'); selected=('qwen25','qwen3vl')
    vmax=float(miss[miss.model.isin(selected)].failing_roots_at_coordinate.max())
    fig,axs=plt.subplots(2,2,figsize=(163/25.4,140/25.4),layout='constrained')
    for row,m in enumerate(selected):
        for col,sl in enumerate(('diagonal','maximum_corner')):
            q=miss[miss.model.eq(m)&miss.slice.eq(sl)]; z=q.pivot(index='image_strength',columns='text_strength',values='failing_roots_at_coordinate').to_numpy()
            ax=axs[row,col]; im=ax.imshow(z,cmap=SEQ,vmin=0,vmax=max(1,vmax))
            ax.set(xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5),xlabel='Text severity',ylabel='Image severity',title=f"{NAMES[m]} · {sl.replace('_',' ')}\n{int(q.missed_roots.iloc[0])} missed questions")
            for i,j in np.ndindex(z.shape): ax.text(j,i,str(int(z[i,j])),ha='center',va='center',color='white' if z[i,j]>.57*vmax else INK)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.75,label='Distinct missed questions failing at this coordinate')
    save(fig,'05_slice_missed_locations','Each map counts only roots missed by that slice; a root can occur in multiple cells.')

    curves=pd.read_csv(a.data/'budget_curves.csv')
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,130/25.4),layout='constrained',sharex=True,sharey=True)
    for ax,m in zip(axs.flat,MODELS):
        for policy,color,label in (('uniform_expected',GRAY,'Uniform (exact expectation)'),('maximum_first',RUST,'Maximum-first'),('diagonal_first',BLUE,'Diagonal-first')):
            q=curves[curves.model.eq(m)&curves.policy.eq(policy)]
            ax.plot(q.budget,100*q.discovery_fraction,color=color,lw=1.1,label=label)
        ax.set(xlim=(0,400),ylim=(0,103),xticks=[0,100,200,400],xlabel='Joint-condition budget',ylabel='Affected questions discovered (%)',title=f"{NAMES[m]} · {int(s.loc[m,'affected_strict41_roots'])} affected")
        ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=5.5,loc='lower right')
    save(fig,'06_fixed_grid_budget_discovery','Observed deterministic policies and exact uniform expectations on the same strict41 fixed-grid cohort, not heldout learned-policy evaluation.')

    rep=pd.read_csv(a.data/'model_replacement.csv'); rep=rep[rep.cohort.eq('old_full_suite_pass')].copy()
    fig,ax=plt.subplots(figsize=(183/25.4,140/25.4),layout='constrained'); left=np.zeros(len(rep))
    for key,color,label in (('clean_regression',RUST,'Clean regression'),('single_regression','#cf9c61','Single-stage regression'),('joint_regression',BLUE,'Joint-stage regression'),('preserved',TEAL,'Entire suite preserved')):
        values=100*rep[key].to_numpy()/rep.denominator.to_numpy(); ax.barh(np.arange(len(rep)),values,left=left,color=color,label=label,height=.7)
        for i,v in enumerate(values):
            if v>=7: ax.text(left[i]+v/2,i,str(int(rep.iloc[i][key])),ha='center',va='center',fontsize=6,color='white' if color!= '#cf9c61' else INK)
        left+=values
    ax.set(yticks=np.arange(len(rep)),yticklabels=[f'{NAMES[r.old_model]} → {NAMES[r.new_model]} (n={int(r.denominator)})' for r in rep.itertuples()],xlim=(0,100),xlabel='Reference model full-suite-success questions (%)',title='Preservation under model replacement · original benchmark')
    ax.invert_yaxis(); ax.legend(ncol=2,fontsize=6,loc='lower center',bbox_to_anchor=(.5,1.04))
    save(fig,'07_reference_success_preservation','Retain the old full-suite-pass population and partition replacement outcomes by first failing stage; not trained repair-method results.')

    allcells=cells[cells.stratum.eq('all')]
    maxex=float(np.nanmax(np.abs(100*allcells.normalized_product_excess)))
    for m in MODELS:
        for column,kind in (('task_accuracy','accuracy'),('normalized_product_excess','product_excess')):
            matrix=np.empty((20,20))
            for r in allcells[allcells.model.eq(m)].itertuples():
                matrix[IMAGES.index(r.image_op)*4+r.image_strength-1,TEXTS.index(r.text_op)*4+r.text_strength-1]=100*getattr(r,column)
            fig,ax=plt.subplots(figsize=(183/25.4,176/25.4),layout='constrained')
            kwargs=dict(cmap='Blues',vmin=0,vmax=100) if kind=='accuracy' else dict(cmap=DIV,norm=TwoSlopeNorm(vmin=-maxex,vcenter=0,vmax=maxex))
            im=ax.imshow(matrix,interpolation='nearest',**kwargs)
            ax.set(xticks=range(20),xticklabels=[f'{t} {i}' for t in TL for i in range(1,5)],yticks=range(20),yticklabels=[f'{u} {i}' for u in IL for i in range(1,5)],xlabel='Text corruption and severity',ylabel='Image corruption and severity',title=f'{NAMES[m]} · new binary {kind.replace("_"," ")}\n1500 questions · 1065 original images')
            ax.tick_params(axis='x',rotation=90,labelsize=5.7,length=0); ax.tick_params(axis='y',labelsize=5.7,length=0)
            for pos in (3.5,7.5,11.5,15.5): ax.axhline(pos,color=INK,lw=.7); ax.axvline(pos,color=INK,lw=.7)
            cb=fig.colorbar(im,ax=ax,shrink=.75,label='Task accuracy (%)' if kind=='accuracy' else '(image×text/clean − joint)/clean, percentage points')
            if kind=='product_excess':
                # Keep diverging-colorbar ticks inside the actual shared domain.
                ticks=np.linspace(-maxex,maxex,5)
                cb.set_ticks(ticks,labels=[f'{v:.2f}' for v in ticks])
            save(fig,f'08_{m}_complete_{kind}_atlas','Complete25 type-pair×16 severity cells; same scale across models. Product excess is a descriptive multiplicative reference, not a causal interaction estimate.')
    with (a.output/'figure_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(dict(backend='Python/matplotlib',model_calls=0,source_tables=str(a.data),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=outputs,primary='task_pass; semantic and format retained separately in accuracy_axes.csv',historical_scores_mixed=False),f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(figures=len(outputs),output=str(a.output))),flush=True)

if __name__=='__main__': main()
