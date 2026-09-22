"""Frozen native-input perturbation audit; see results/characterization/PROTOCOL.md."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model.smq import SMQModel
from src.model.motion_quantizer import euclidean_dist

SEEDS = [1538574472, 111, 222]
OUT = ROOT / 'results/characterization'


def trajectory(x, dataset):
    # Native C,T,V,M -> T,C,V, preserving cache coordinate convention.
    if dataset == 'lara':
        return x[3:6, :, :21, 0].transpose(1, 2, 0).reshape(x.shape[1], -1)
    return x[:, :, :, 0].transpose(1, 0, 2).reshape(x.shape[1], -1)


def changes(a, b, scale, fps, distance_scales):
    a, b = a.astype(np.float64) / scale, b.astype(np.float64) / scale
    disp_a, disp_b = a-a[:1], b-b[:1]
    va, vb = np.diff(a, axis=0)*fps, np.diff(b, axis=0)*fps
    va, vb = np.concatenate([va[:1], va]), np.concatenate([vb[:1], vb])
    posture = float(np.sqrt(np.mean((a.mean(0)-b.mean(0))**2)))
    displacement = float(np.sqrt(np.mean((disp_a-disp_b)**2)))
    velocity = float(np.sqrt(np.mean((va-vb)**2)))
    return dict(posture_change=posture, displacement_change=displacement,
                velocity_change=velocity,
                composite_change=float(.5*(displacement/distance_scales[0]+velocity/distance_scales[1])))


def encode(model, crops, device, width):
    result=[]
    with torch.inference_mode():
        for i in range(0,len(crops),16):
            x = torch.tensor(np.stack(crops[i:i+16]), dtype=torch.float32, device=device)
            n,c,t,v,m=x.shape
            packed=x.permute(0,4,3,1,2).contiguous().view(n*m*v,c,t)
            latent=model.encoder(packed,torch.ones_like(packed))
            latent=latent.view(n*m,v,model.latent_dim,t).permute(0,3,1,2)
            latent=latent.reshape(n,m,t,v,model.latent_dim).permute(0,2,3,1,4).reshape(n,t,-1)
            # Quantize exactly the represented interval, not the crop-origin grid.
            central=latent[:,14:14+width]
            result.extend(euclidean_dist(central,model.vq._embedding).argmin(1).cpu().tolist())
    return np.asarray(result)


def interval(values, subjects):
    unique=np.unique(subjects)
    sums=np.array([np.sum(values[subjects==s]) for s in unique])
    counts=np.array([np.sum(subjects==s) for s in unique])
    rng=np.random.default_rng(20260921)
    pick=rng.integers(0,len(unique),(2000,len(unique)))
    draws=sums[pick].sum(1)/counts[pick].sum(1)
    return list(map(float,np.quantile(draws,[.025,.975])))


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def balanced_summary(rows, k):
    """Equal weight over represented original codes, participant-cluster CI."""
    subjects=sorted(set(r['subject'] for r in rows))
    lookup={s:i for i,s in enumerate(subjects)}
    counts=np.zeros((len(subjects),k))
    for r in rows: counts[lookup[r['subject']],r['original_code']]+=1
    picks=np.random.default_rng(20260921).integers(0,len(subjects),(2000,len(subjects)))
    denominator=counts[picks].sum(1)
    result=dict(dataset=rows[0]['dataset'],seed=rows[0]['seed'],perturbation=rows[0]['perturbation'],represented_codes=int((counts.sum(0)>0).sum()),participants=len(subjects),n=len(rows))
    for metric in ['agreement','posture_change','displacement_change','velocity_change','composite_change']:
        totals=np.zeros_like(counts)
        for r in rows: totals[lookup[r['subject']],r['original_code']]+=r[metric]
        percode=np.divide(totals.sum(0),counts.sum(0),out=np.full(k,np.nan),where=counts.sum(0)>0)
        boot=np.divide(totals[picks].sum(1),denominator,out=np.full_like(denominator,np.nan),where=denominator>0)
        lo,hi=np.quantile(np.nanmean(boot,axis=1),[.025,.975])
        result.update({metric:float(np.nanmean(percode)),metric+'_ci_low':float(lo),metric+'_ci_high':float(hi)})
    return result


def run(dataset):
    with np.load(OUT/f'{dataset}_cache.npz',allow_pickle=True) as packed:
        cache={key:packed[key] for key in packed.files}
    width=50 if dataset=='lara' else 60
    fps=50 if dataset=='lara' else 60
    k=8 if dataset=='lara' else 10
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(.5)
    torch.set_num_threads(4)
    models=[]
    for seed in SEEDS:
        model=SMQModel(in_channels=6,filters=128,num_layers=3,latent_dim=16,
                       num_actions=k,num_joints=22 if dataset=='lara' else 6,patch_size=width)
        path=ROOT/f'models/exp/T1/{dataset}/K{k}_s{seed}/{dataset}/epoch-30.model'
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
        models.append(model.to(device).eval())
    rows=[]
    validation={'dataset':dataset,'clean_crop_checks':0,'clean_crop_mismatches':0,
                'raw_overlap_max_abs':0.,'cache_trajectory_max_abs':0.,'shift_ineligible_patches':0}
    selected=[]
    for rec in np.unique(cache['rec'][cache['evaluation']]):
        eligible=np.flatnonzero(cache['evaluation'] & (cache['rec']==rec))
        ix=np.linspace(0,len(eligible)-1,min(6,len(eligible))).astype(int)
        selected.extend(eligible[ix].tolist())
    for rnum,rec in enumerate(np.unique(cache['rec'][selected])):
        native=np.load(ROOT/'data'/dataset/'features'/str(cache['names'][rec]))
        ids=[p for p in selected if cache['rec'][p]==rec]
        crops=[]; metadata=[]
        for p in ids:
            start=int(cache['start'][p]); clean=native[:,start-14:start+width+14].copy()
            if clean.shape[1]!=width+28: raise ValueError('Incomplete original context')
            base=trajectory(clean[:,14:14+width],dataset)
            err=float(np.max(np.abs(base-cache['trajectory'][p])))
            validation['cache_trajectory_max_abs']=max(validation['cache_trajectory_max_abs'],err)
            if not np.array_equal(base.astype(np.float32),cache['trajectory'][p]):
                raise ValueError(f'Trajectory cache convention mismatch beyond float32 storage rounding {dataset}: {err}')
            variants={'clean':clean}
            noise_diagnostics={}
            if start+width+16<=native.shape[1]:
                variants['shift_2']=native[:,start-12:start+width+16].copy()
            else: validation['shift_ineligible_patches']+=1
            for amplitude in [.001,.005]:
                # Hu native storage is integer; perturb in floating point before
                # the model's native float32 conversion, never truncate noise.
                perturb=clean.astype(np.float64,copy=True)
                key=f'{dataset}/{cache["names"][rec]}/{start}/{amplitude}'
                rng=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
                if dataset=='lara':
                    noise=rng.normal(size=perturb[3:6].shape)*float(cache['noise_scale'])*amplitude
                    drawn_rms=float(np.sqrt(np.mean(noise**2))/float(cache['noise_scale']))
                    noise-=noise[:,:,21:22,:]
                    perturb[3:6]+=noise.astype(perturb.dtype)
                    effective_rms=float(np.sqrt(np.mean((perturb[3:6]-clean[3:6])**2))/float(cache['noise_scale']))
                    if np.any(perturb[3:6,:,21,:]!=0): raise ValueError('Root is not zero')
                    if not np.array_equal(perturb[:3],clean[:3]): raise ValueError('Orientation changed')
                else:
                    noise=rng.normal(size=perturb.shape)*cache['noise_scale'][:,None,:,None]*amplitude
                    drawn_rms=float(np.sqrt(np.mean((noise/cache['noise_scale'][:,None,:,None])**2)))
                    perturb+=noise.astype(perturb.dtype)
                    effective_rms=float(np.sqrt(np.mean(((perturb-clean)/cache['noise_scale'][:,None,:,None])**2)))
                variants[f'noise_{amplitude}']=perturb
                noise_diagnostics[f'noise_{amplitude}']=dict(requested_independent_noise_rms_ratio=amplitude,drawn_independent_noise_rms_ratio=drawn_rms,effective_input_noise_rms_ratio=effective_rms)
            for label,crop in variants.items():
                after=trajectory(crop[:,14:14+width],dataset)
                desc=changes(base,after,cache['posture_scale'],fps,cache['distance_scales'])
                overlap={f'overlap_{key}':value for key,value in changes(base[2:],after[:-2],cache['posture_scale'],fps,cache['distance_scales']).items()} if label=='shift_2' else {f'overlap_{key}':float('nan') for key in desc}
                if label=='shift_2':
                    err=float(np.max(np.abs(base[2:]-after[:-2])))
                    validation['raw_overlap_max_abs']=max(err,validation['raw_overlap_max_abs'])
                    if err!=0 or any(v!=0 for v in overlap.values()): raise ValueError('Common raw frames disagree')
                diagnostics=noise_diagnostics.get(label,dict(requested_independent_noise_rms_ratio=0.,drawn_independent_noise_rms_ratio=0.,effective_input_noise_rms_ratio=0.))
                metadata.append(dict(patch=int(p),recording=str(cache['names'][rec]),subject=str(cache['subjects'][rec]),start=start,perturbation=label,**desc,**overlap,**diagnostics))
                crops.append(crop)
        for si,model in enumerate(models):
            codes=encode(model,crops,device,width)
            for meta,code in zip(metadata,codes):
                original=int(cache['codes'][meta['patch'],si])
                if meta['perturbation']=='clean':
                    validation['clean_crop_checks']+=1
                    if code!=original:
                        validation['clean_crop_mismatches']+=1
                        (OUT/f'stability_{dataset}_validation.json').write_text(json.dumps(validation,indent=2))
                        raise ValueError(f'Clean context assignment mismatch {dataset} {meta} seed {SEEDS[si]}: {code} != {original}')
                else:
                    rows.append(dict(dataset=dataset,seed=SEEDS[si],**meta,original_code=original,perturbed_code=int(code),agreement=int(code==original)))
        if rnum%25==0: print(f'{dataset}: {rnum+1} recordings complete',flush=True)
    write_csv(OUT/f'stability_{dataset}_patches.csv',rows)
    summary=[]; percode=[]; balanced=[]
    for seed in SEEDS:
        for perturbation in ['shift_2','noise_0.001','noise_0.005']:
            rr=[r for r in rows if r['seed']==seed and r['perturbation']==perturbation]
            balanced.append(balanced_summary(rr,k))
            subjects=np.array([r['subject'] for r in rr])
            for code in [None]+list(range(k)):
                sub=rr if code is None else [r for r in rr if r['original_code']==code]
                entry=dict(dataset=dataset,seed=seed,perturbation=perturbation,code='all' if code is None else code,n=len(sub),frequency=len(sub)/len(rr),participants=len(set(r['subject'] for r in sub)))
                for metric in ['agreement','posture_change','displacement_change','velocity_change','composite_change','overlap_posture_change','overlap_displacement_change','overlap_velocity_change','overlap_composite_change']:
                    values=np.array([r[metric] for r in sub])
                    entry[metric]=float(values.mean()) if len(values) else None
                    bounds=interval(values,np.array([r['subject'] for r in sub])) if len(values) and np.isfinite(values).all() else [None,None]
                    entry[metric+'_ci_low'],entry[metric+'_ci_high']=bounds
                (summary if code is None else percode).append(entry)
    write_csv(OUT/f'stability_{dataset}_summary.csv',summary)
    write_csv(OUT/f'stability_{dataset}_per_code.csv',percode)
    write_csv(OUT/f'stability_{dataset}_code_balanced.csv',balanced)
    spread=[]
    for perturbation in ['shift_2','noise_0.001','noise_0.005']:
        values=[r['agreement'] for r in summary if r['perturbation']==perturbation]
        spread.append(dict(perturbation=perturbation,agreement_seed_mean=float(np.mean(values)),agreement_seed_sd=float(np.std(values,ddof=1))))
    validation.update(selected_patches=len(selected),selected_recordings=len(np.unique(cache['rec'][selected])),seed_spread=spread,bootstrap='2000 participant-cluster draws, weighted by patch count; seed SD separate',descriptor_scope='root-relative joint xyz' if dataset=='lara' else 'sensor signal analogues, not posture or joint movement')
    validation['noise_diagnostics']={label:{key:float(np.mean([r[key] for r in rows if r['seed']==SEEDS[0] and r['perturbation']==label])) for key in ['requested_independent_noise_rms_ratio','drawn_independent_noise_rms_ratio','effective_input_noise_rms_ratio']} for label in ['noise_0.001','noise_0.005']}
    validation['noise_diagnostic_note']='RMS over complete model context, divided by calibration scale; LARa effective input includes root subtraction, zero root and all22 joints; Hu normalized per channel/sensor. Perturbations evaluated in float64 before native model float32 cast.'
    (OUT/f'stability_{dataset}_validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation),flush=True)


if __name__=='__main__':
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['lara','hugadb'])
    run(ap.parse_args().dataset)
