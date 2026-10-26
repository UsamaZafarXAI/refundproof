"""Source-group evaluation, calibration and operating-policy analysis."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import sys, json, time, random, hashlib, io, traceback, shutil, zipfile, platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from scipy.stats import ttest_1samp, beta
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator,ClassifierMixin
import joblib

FEATURES = ['blur_var','brightness','contrast','noise_sigma','lbp_entropy',
 'dct_ac_energy','dct_dc_std','dct_hf_kurtosis','dct_energy_ratio','fft_slope',
 'fft_hf_ratio','cc_mean','grad_mean','grad_entropy','noise_residual_var','dct_skewness']
KNOWN = ['gpt-image-2','grok-imagine-image','nano-banana-2','qwen-image-2.0-pro','qwen-image-edit-max','wan2.7-image-pro']
SEEDS = [42,123,777]
ROOT = Path(os.environ.get('WORK_DIR','/work/refundproof'))
OUT = ROOT/'results'
CONFIG = {'version':'FraudBench-1.0','dataset':'TristanYan/FraudBench',
 'seeds':SEEDS,'folds':3,'epochs':8,'patience':2,'batch_size':32,
 'learning_rate':0.0001,'weight_decay':0.0001,'bootstrap':500,
 'jpeg':[95,75,50],'fpr_targets':[0.01,0.05,0.10],
 'cost_ratios':[2,5,10,20],'prevalences':[0.01,0.05,0.10,0.25,0.50],
 'calibration':'group-disjoint held-out sigmoid (shallow), temperature (M5)',
 'primary_group':'source review merged by decoded-image equality and pHash distance <= 4',
 'm5':'ImageNet EfficientNet-B0; last two feature blocks and classifier fine-tuned',
 'external_case_validation':False,
 'ufd_author_revision':'030495aea3300a8b54c0ec37ec7fe1dd7e63c619'}
DEADLINE = time.monotonic()+max(1800,float(os.environ.get('MAX_HOURS','24'))*3600-3600)
DATA = None
API = None
REPO = os.environ.get('OUTPUT_REPO','')
PERSIST=Path(os.environ['PERSIST_DIR']) if os.environ.get('PERSIST_DIR') else None
SYNCED={}

class BudgetStop(Exception): pass
def budget():
    if time.monotonic()>DEADLINE: raise BudgetStop('Time allowance reached; resume the same repository to finish.')
def say(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
    if OUT.exists():
        with (OUT/'run.log').open('a',encoding='utf-8') as f: f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+s+'\n')
def write_json(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8'); tmp.replace(p)
def hash_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def stable(s): return hashlib.sha256(str(s).encode()).hexdigest()[:24]
def reset(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
def sync():
    if PERSIST is not None:
        dest=PERSIST/'results'; dest.mkdir(parents=True,exist_ok=True)
        for p in OUT.rglob('*'):
            if not p.is_file() or p.suffix=='.tmp': continue
            target=dest/p.relative_to(OUT); target.parent.mkdir(parents=True,exist_ok=True)
            key=p.relative_to(OUT).as_posix(); stamp=(p.stat().st_size,p.stat().st_mtime_ns)
            if target.exists() and (SYNCED.get(key)==stamp or (target.stat().st_size,target.stat().st_mtime_ns)==stamp):
                SYNCED[key]=stamp; continue
            tmp=target.with_name(target.name+'.copying'); shutil.copy2(p,tmp); tmp.replace(target)
            SYNCED[key]=stamp
        return
    if API is None or not REPO: return

    API.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(OUT),path_in_repo='results',
        ignore_patterns=['*.tmp','*.zip'],commit_message='Save completed experiments and provenance')
def load_image(path,quality=None):
    with Image.open(path) as src: im=ImageOps.exif_transpose(src).convert('RGB')
    if quality is not None:
        buf=io.BytesIO(); im.save(buf,format='JPEG',quality=int(quality),subsampling=2)
        buf.seek(0); im=Image.open(buf).convert('RGB'); im.load()
    return im

def split_groups(y,g,seed):
    """Disjoint core / early-stop / calibration groups, all with both classes."""
    y=np.asarray(y); g=np.asarray(g)
    def hold(idx,frac,s):
        for a,b in GroupShuffleSplit(n_splits=300,test_size=frac,random_state=s).split(idx,y[idx],g[idx]):
            a,b=idx[a],idx[b]
            if len(np.unique(y[a]))==2 and len(np.unique(y[b]))==2: return a,b
        raise ValueError('Insufficient independent groups for a two-class inner split')
    rest,cal=hold(np.arange(len(y)),0.20,seed)
    core,val=hold(rest,0.20,seed+1)
    assert not (set(g[core])&set(g[val]) or set(g[core])&set(g[cal]) or set(g[val])&set(g[cal]))
    return core,val,cal
def outer(y,g,seed):
    out=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
    for tr,te in out:
        assert not set(np.asarray(g)[tr])&set(np.asarray(g)[te])
        if len(np.unique(np.asarray(y)[tr]))!=2 or len(np.unique(np.asarray(y)[te]))!=2:
            raise ValueError('Outer fold has only one class; inspect grouping and class counts.')
    return out
def temperature(z,y):
    z=np.asarray(z,dtype=float); y=np.asarray(y,dtype=int)
    r=minimize_scalar(lambda lt:log_loss(y,expit(z/np.exp(lt)),labels=[0,1]),bounds=(-4,4),method='bounded')
    if not r.success: raise RuntimeError('Temperature optimization failed')
    return float(np.exp(r.x))
def ece(y,p,bins=10):
    idx=np.minimum((np.asarray(p)*bins).astype(int),bins-1)
    return float(sum((idx==b).mean()*abs(np.mean(np.asarray(y)[idx==b])-np.mean(np.asarray(p)[idx==b])) for b in range(bins) if (idx==b).any()))
def metrics(y,p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-7,1-1e-7); pred=p>=.5
    return {'n':len(y),'real':int((y==0).sum()),'fake':int((y==1).sum()),
      'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
      'ap':float(average_precision_score(y,p)) if y.sum() else np.nan,
      'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
      'ece_10':ece(y,p), 'accuracy':float((pred==y).mean()),
      'fpr':float(pred[y==0].mean()) if (y==0).any() else np.nan,
      'tpr':float(pred[y==1].mean()) if (y==1).any() else np.nan}
def thresholds(y,p):
    """Exact ROC candidates including reject-all, retaining tied scores."""
    f,t,h=roc_curve(y,p,drop_intermediate=False)
    return np.asarray(f),np.asarray(t),np.asarray(h)
def fpr_threshold(y,p,target):
    f,t,h=thresholds(y,p); eligible=np.flatnonzero(f<=target)

    j=eligible[np.argmax(t[eligible])]
    return float(h[j])
def rates(y,p,h):
    y=np.asarray(y); pred=np.asarray(p)>=h; n=int((y==0).sum()); m=int((y==1).sum())
    fp=int(pred[y==0].sum()); tp=int(pred[y==1].sum()); fn=m-tp
    def ci(k,n):
        if not n: return (np.nan,np.nan)
        return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.)
    lo,hi=ci(fp,n)
    return {'threshold':h,'false_positives':fp,'false_negatives':fn,'real_n':n,'fake_n':m,
      'fpr':fp/n if n else np.nan,'tpr':tp/m if m else np.nan,
      'fpr_binomial_lo':lo,'fpr_binomial_hi':hi,
      'interval_note':'Exact binomial descriptive interval; images within source groups are dependent. See grouped bootstrap.'}
def operational_rows(cal_y,cal_p,test_y,test_p,tag):
    out=[]
    for target in CONFIG['fpr_targets']:
        h=fpr_threshold(cal_y,cal_p,target)
        out.append(dict(tag,policy='calibration_FPR_target',target=target,selection_fpr=rates(cal_y,cal_p,h)['fpr'],**rates(test_y,test_p,h)))
    f,t,h=thresholds(cal_y,cal_p)

    for ratio in CONFIG['cost_ratios']:
        for prev in CONFIG['prevalences']:
            j=int(np.argmin(ratio*(1-prev)*f+prev*(1-t)))
            rr=rates(test_y,test_p,h[j])
            out.append(dict(tag,policy='minimum_calibration_expected_cost',cost_fp_to_fn=ratio,prevalence=prev,
                expected_test_cost=ratio*(1-prev)*rr['fpr']+prev*(1-rr['tpr']),**rr))
    for lo,hi in [(.10,.90),(.20,.80),(.30,.70)]:
        keep=(test_p<=lo)|(test_p>=hi)
        out.append(dict(tag,policy='fixed_review_band',lower=lo,upper=hi,coverage=float(keep.mean()),
           selective_error=float(((test_p[keep]>=.5)!=test_y[keep]).mean()) if keep.any() else np.nan))
    return out
def operating_uncertainty(rows,y,p,g):

    rng=np.random.default_rng(42); unique=np.unique(g); ix={v:np.flatnonzero(g==v) for v in unique}
    indexes=[np.concatenate([ix[v] for v in rng.choice(unique,len(unique))]) for _ in range(200)]
    for row in rows:
        if row['policy']!='calibration_FPR_target': continue
        vals=[]
        for ids in indexes:
            yy=y[ids]; pp=p[ids]>=row['threshold']
            if (yy==0).any() and (yy==1).any(): vals.append([pp[yy==0].mean(),pp[yy==1].mean()])
        a=np.asarray(vals)
        for j,name in enumerate(['fpr','tpr']):
            row[name+'_group_bootstrap_lo']=float(np.quantile(a[:,j],.025)); row[name+'_group_bootstrap_hi']=float(np.quantile(a[:,j],.975))
    return rows
def holm(p):
    p=np.asarray(p); order=np.argsort(p); r=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]); out=np.empty(len(p)); out[order]=np.minimum(1,r); return out

def audit_dataset(root):
    import cv2
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp'] and not p.name.startswith('.'))
    edits={}; issues=[]; raw=[]
    for p in root.glob('*/DeepFake/Metadata/*.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        for a in d.get('per_image',[]):
            key=(p.parts[-4],str(d.get('source_review','')),a.get('image',''))
            edits[key]=(d,a,p.relative_to(root).as_posix())
    cache={}
    for p in files:
        budget(); rel=p.relative_to(root); parts=rel.parts
        if len(parts)<4 or parts[1] not in ['Negative','Positive','DeepFake']: continue
        category,bucket=parts[:2]; gen=parts[2] if bucket=='DeepFake' else 'real'; review=parts[-2]
        if not review.startswith('Review_'): continue
        source_bucket='Positive' if bucket=='DeepFake' else bucket
        source_dir=root/category/source_bucket/review
        mp=next(iter(sorted(source_dir.glob('MetaReview*.json'))),None)
        if mp not in cache: cache[mp]=json.loads(mp.read_text(encoding='utf-8')) if mp else {}
        md=cache[mp]; source_review=str(md.get('review_id') or category+'/'+source_bucket+'/'+review)
        edit=edits.get((category,review,p.name)) if bucket=='DeepFake' else None
        if bucket=='DeepFake' and edit is None: raise ValueError('No source/edit metadata for '+str(rel))
        source_path=root/category/'Positive'/review/p.name if bucket=='DeepFake' else p
        if not source_path.exists(): raise ValueError('Missing paired source '+str(source_path))
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB'); width,height=im.size
            rgbhash=hashlib.sha256(np.asarray(im).tobytes()+str(im.size).encode()).hexdigest()
            small=np.array(im.resize((32,32)).convert('L'),dtype=np.float32)
            dct=cv2.dct(small)[:8,:8].flatten(); bits=dct>np.median(dct[1:]); ph=int.from_bytes(np.packbits(bits).tobytes(),'big')
        text=(edit[1].get('reviewer_comment','') if edit else str(md.get('text') or md.get('review_text') or ''))
        raw.append({'image_id':stable(rel.as_posix()),'path':str(p),'relative_path':rel.as_posix(),
          'category':category,'bucket':bucket,'generator':gen,'review_folder':review,
          'source_review_key':stable(source_review),'artifact_id':stable(str(source_path.relative_to(root))),
          'source_path':str(source_path),'source_relative_path':source_path.relative_to(root).as_posix(),
          'product_key':stable(md.get('parent_asin') or md.get('asin') or source_review),
          'original_review_id_available':bool(md.get('review_id')),'text':text,
          'text_origin':'generated_reviewer_comment' if bucket=='DeepFake' else 'original_review',
          'damage':edit[1].get('damage','unknown') if edit else 'unknown',
          'matched_label':edit[1].get('damage','unknown') if edit else 'unknown',
          'label_origin':'FraudBench directory annotation; not verified fraud outcome',
          'metadata_path':edit[2] if edit else (mp.relative_to(root).as_posix() if mp else ''),
          'label':1 if bucket=='DeepFake' else 0,'eligible':bucket!='Positive',
          'sha256':hash_file(p),'pixel_sha256':rgbhash,'phash':ph,'width':width,'height':height})
    df=pd.DataFrame(raw)
    if df.empty: raise ValueError('FraudBench directory layout was not found')

    parent=list(range(len(df)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b): parent[find(a)]=find(b)
    for col in ['source_review_key','pixel_sha256']:
        for idx in df.groupby(col).indices.values():
            for j in idx[1:]: union(int(idx[0]),int(j))
    representatives={}
    for j,h in enumerate(df.phash):
        if h in representatives: union(j,representatives[h])
        else: representatives[h]=j
    hs=list(representatives.items())
    for k,(h,j) in enumerate(hs):
        if k%100==0: budget()
        for other,oi in hs[:k]:
            if (int(h)^int(other)).bit_count()<=4: union(j,oi)
    df['group']=[stable('group/'+df.iloc[find(i)].source_review_key) for i in range(len(df))]

    conflict=set(df[df.eligible].groupby('pixel_sha256').label.nunique().loc[lambda x:x>1].index)
    df['exclusion_reason']=np.where(df.bucket=='Positive','undamaged source; pairing only',np.where(df.pixel_sha256.isin(conflict),'identical pixels have conflicting labels',''))
    df.loc[df.pixel_sha256.isin(conflict),'eligible']=False
    df.to_csv(OUT/'manifest_all.csv',index=False)
    before=df[df.bucket!='Positive']; usable=df[df.eligible].copy().reset_index(drop=True)
    for name,frame in [('before',before),('after',usable)]:
        frame.groupby(['category','bucket','generator'],dropna=False).size().rename('n').reset_index().to_csv(OUT/f'domain_distribution_{name}.csv',index=False)
    nconf=int((~df.eligible & (df.bucket!='Positive')).sum())
    write_json(OUT/'stage_A_provenance.json',{'downloaded_image_files':len(files),'recognized_images':len(df),
      'main_before_exclusions':len(before),'main_after_exclusions':len(usable),'undamaged_pairing_images':int((df.bucket=='Positive').sum()),
      'bucket_counts':df.bucket.value_counts().to_dict(),'generator_counts':usable.generator.value_counts().to_dict(),
      'source_groups':int(usable.group.nunique()),'duplicate_label_conflict_exclusions':nconf,
      'original_review_id_coverage':float(usable.original_review_id_available.mean()),
      'class_prevalence':float(usable.label.mean()),'near_duplicate_rule':'64-bit pHash Hamming <=4; connected components',
      'license':'CC-BY-NC-SA-4.0; retain author attribution; no commercial-use authorization',
      'mask_generation':'Existing author-provided edits; no new masks or generation. No localization mask ground truth asserted.',
      'matched_label':'Synthetic damage from Edit metadata; genuine damage categories not inferred from review text.',
      'warnings':['Review grouping is conservative, not proof of absence of all semantic duplicates.',
         'FraudBench uses curated product reviews, not adjudicated investigative case files.',
         'Generated comments and original comments differ in provenance; M4 requires T0 and text-coverage analysis.']})
    if usable.label.nunique()!=2: raise ValueError('Both classes are required')
    if usable.groupby('group').size().max()>len(usable)*.20: raise ValueError('One duplicate component exceeds 20% of data; manual provenance inspection required.')
    write_json(OUT/'text_coverage.json',usable.assign(has_text=usable.text.str.strip().str.len()>0).groupby(['bucket','text_origin']).has_text.agg(['count','mean']).reset_index().to_dict('records'))
    return df,usable

def forensic_one(arg):
    path,quality=arg
    from legacy_features import _extract_features_worker
    temp=None
    if quality is not None:
        temp=ROOT/'temporary'/f'{stable(path)}_{quality}.jpg'; temp.parent.mkdir(exist_ok=True)
        load_image(path,quality).save(temp,format='PNG')
        path=str(temp)
    r=_extract_features_worker(('unused','unused',path,path))
    if temp: temp.unlink(missing_ok=True)
    if not r.get('ok'): raise RuntimeError('Forensic extraction failed: '+str(r))
    return [r['real_'+f] for f in FEATURES]
def get_forensic(frame,quality=None):
    import cv2
    cv2.setNumThreads(1)
    p=OUT/'cache'/f'forensic_{quality or "original"}.npy'; p.parent.mkdir(exist_ok=True)
    if p.exists(): return np.load(p)
    say(f'Extracting 16 forensic features for {len(frame)} images, JPEG={quality}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        arr=np.array(list(pool.map(forensic_one,[(p,quality) for p in frame.path])),dtype=np.float32)
    arr[~np.isfinite(arr)]=np.nan
    np.save(p,arr); return arr
def embeddings(frame):
    import torch
    import torchvision.models as tm
    from transformers import CLIPModel,CLIPProcessor
    from huggingface_hub import snapshot_download
    paths=list(frame.path); result={}
    for name in ['efn','clip']:
        target=OUT/'cache'/f'{name}.npy'
        if target.exists(): result[name]=np.load(target); continue
        budget(); reset(42)
        if name=='efn':
            w=tm.EfficientNet_B0_Weights.IMAGENET1K_V1; model=tm.efficientnet_b0(weights=w)
            model.classifier=torch.nn.Identity(); transform=w.transforms()
        else:
            rev=CONFIG.get('clip_revision') or API.model_info('openai/clip-vit-base-patch32').sha
            CONFIG['clip_revision']=rev; write_json(OUT/'configuration.json',CONFIG)
            mp=snapshot_download('openai/clip-vit-base-patch32',revision=rev,allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin'])
            model=CLIPModel.from_pretrained(mp); processor=CLIPProcessor.from_pretrained(mp)
        model=model.cuda().eval(); chunks=[]
        with torch.inference_mode():
            for start in range(0,len(paths),32):
                budget(); ims=[load_image(p) for p in paths[start:start+32]]
                if name=='efn': vec=model(torch.stack([transform(im) for im in ims]).cuda())
                else:
                    inp=processor(images=ims,return_tensors='pt'); vec=model.get_image_features(pixel_values=inp['pixel_values'].cuda())
                    vec=vec/vec.norm(dim=1,keepdim=True).clamp_min(1e-8)
                chunks.append(vec.cpu().numpy())
                if start%1024==0: say(f'{name} embeddings {start}/{len(paths)}')
        result[name]=np.concatenate(chunks); np.save(target,result[name]); del model
        torch.cuda.empty_cache(); sync()
    return result

def model_input(name,F,E,C,text):
    if name=='B1': return F[:,[9,10]]
    if name=='B2': return F[:,[3,14]]
    if name=='E0': return E
    if name=='C0': return C
    if name=='B3': return np.hstack([F,E])
    if name in ['M3','M4']: a=np.hstack([F,C])
    else: a=F
    if name in ['T0','M4']:
        d=pd.DataFrame(a); d.columns=[str(i) for i in d.columns]; d['text']=text
        return d[['text']] if name=='T0' else d
    return a

def ufd_logits(frame):
    """Pretrained UniversalFakeDetect head; equivalent HF OpenAI CLIP backbone.
    Use the author's CenterCrop(224) preprocessing (no resize, no feature normalization).
    Raw baseline is not fitted to FraudBench; group-disjoint calibration is reported separately.
    """
    import torch, urllib.request
    from torchvision import transforms as T
    from transformers import CLIPModel
    from huggingface_hub import snapshot_download
    cache=OUT/'cache'/'ufd_logits.npy'; cache.parent.mkdir(exist_ok=True)
    if cache.exists(): return np.load(cache)
    budget(); reset(42)
    url='https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/'+CONFIG['ufd_author_revision']+'/pretrained_weights/fc_weights.pth'
    headfile=OUT/'ufd_pretrained_head.pth'
    if not headfile.exists(): headfile.write_bytes(urllib.request.urlopen(url,timeout=60).read())
    head=torch.nn.Linear(768,1); head.load_state_dict(torch.load(headfile,map_location='cpu',weights_only=True)); head=head.cuda().eval()
    rev=CONFIG.get('ufd_clip_revision') or API.model_info('openai/clip-vit-large-patch14').sha
    CONFIG['ufd_clip_revision']=rev; write_json(OUT/'configuration.json',CONFIG)
    mp=snapshot_download('openai/clip-vit-large-patch14',revision=rev,allow_patterns=['*.json','*.safetensors','pytorch_model.bin'])
    model=CLIPModel.from_pretrained(mp).cuda().eval()
    transform=T.Compose([T.CenterCrop(224),T.ToTensor(),T.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])])
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(frame),16):
            budget(); xx=torch.stack([transform(load_image(p)) for p in frame.path.iloc[start:start+16]]).cuda()
            features=model.get_image_features(pixel_values=xx)
            chunks.append(head(features.float()).squeeze(1).cpu().numpy())
            if start%1024==0: say(f'Pretrained UFD {start}/{len(frame)}')
    z=np.concatenate(chunks); np.save(cache,z)
    write_json(OUT/'ufd_provenance.json',{'publication':'Ojha, Li and Lee, CVPR 2023, Towards Universal Fake Image Detectors that Generalize Across Generative Models',
      'author_repo':'https://github.com/WisconsinAIVision/UniversalFakeDetect','author_commit':CONFIG['ufd_author_revision'],
      'head_sha256':hash_file(headfile),'backbone':'openai/clip-vit-large-patch14','backbone_revision':rev,
      'implementation':'Author linear head with equivalent Hugging Face OpenAI CLIP weights, raw projected features, author center-crop preprocessing',
      'adaptation':'Detector weights fixed; sigmoid calibration uses outer-training calibration groups only',
      'caveat':'Original pretrained training-set overlap with public reviews has not been exhaustively audited; not a newest-detector comparison.'})
    del model,head; torch.cuda.empty_cache(); sync(); return z
def take(x,i): return x.iloc[i] if isinstance(x,pd.DataFrame) else x[i]
def estimator(name,x,seed):
    num=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler())])
    if name in ['T0','M4']:
        transformers=[('text',TfidfVectorizer(max_features=3000,ngram_range=(1,2),min_df=1),'text')]
        if name=='M4': transformers.append(('visual',num,[c for c in x.columns if c!='text']))
        prep=ColumnTransformer(transformers,sparse_threshold=1.)
    else: prep=num
    if name=='M1GB': model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1,random_state=seed)
    elif name=='B2': model=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=3,n_jobs=4,random_state=seed)
    elif name=='B1': model=LinearSVC(C=1,random_state=seed,dual='auto')
    else: model=LogisticRegression(C=1,max_iter=2000,random_state=seed)
    return Pipeline([('features',prep),('classifier',model)])
def score(est,x):
    if hasattr(est,'decision_function'): return np.asarray(est.decision_function(x))
    return logit(np.clip(est.predict_proba(x)[:,1],1e-6,1-1e-6))
def sigmoid_fit(z,y):
    return LogisticRegression(C=1e6,max_iter=2000).fit(np.asarray(z).reshape(-1,1),y)
def calibrated(sig,z): return sig.predict_proba(np.asarray(z).reshape(-1,1))[:,1]
class CalibratedView(ClassifierMixin,BaseEstimator):
    def __init__(self,est,sig): self.est=est; self.sig=sig; self.classes_=np.array([0,1])
    def fit(self,x,y): raise RuntimeError('This held-out scorer is already fitted')
    def predict_proba(self,x):
        p=calibrated(self.sig,score(self.est,x)); return np.column_stack([1-p,p])
def export_predictions(ids,raw,p,meta):
    d=DATA.iloc[ids][['image_id','artifact_id','group','category','generator','damage','label']].copy()
    d['row_id']=ids; d['raw_probability']=raw; d['probability']=p
    for k,v in meta.items(): d[k]=v
    return d
def shallow(name,x,tr,te,seed,tag):
    budget(); key=stable(json.dumps(dict(tag,model=name),sort_keys=True)); base=OUT/'experiments'/key
    base.mkdir(parents=True,exist_ok=True); predfile=base/'predictions.csv'
    if (base/'done.json').exists(): return pd.read_csv(predfile)
    reset(seed); y=DATA.label.to_numpy(); g=DATA.group.to_numpy()
    core,val,cal=split_groups(y[tr],g[tr],seed); fit=tr[np.r_[core,val]]; ca=tr[cal]
    if name=='UFD':
        est=None; cz=UFD[ca]; tz=UFD[te]
    else:
        est=estimator(name,x,seed); est.fit(take(x,fit),y[fit]); cz=score(est,take(x,ca)); tz=score(est,take(x,te))
    sig=sigmoid_fit(cz,y[ca]); p=calibrated(sig,tz); cp=calibrated(sig,cz)
    pred=export_predictions(te,expit(tz),p,dict(tag,model=name)); pred.to_csv(predfile,index=False)
    rows=operational_rows(y[ca],cp,y[te],p,dict(tag,model=name))
    if tag['evaluation']=='main' and name=='M1': rows=operating_uncertainty(rows,y[te],p,g[te])
    pd.DataFrame(rows).to_csv(base/'operational.csv',index=False)
    pd.DataFrame({'row_id':ca,'label':y[ca],'raw_logit':cz,'probability':cp}).to_csv(base/'calibration.csv',index=False)
    save_partition(base,fit,[],ca,te)
    joblib.dump((est,sig),base/'model.joblib',compress=3)
    if name=='M1' and tag['evaluation']=='main':
        imp=permutation_importance(CalibratedView(est,sig),take(x,te),y[te],scoring='roc_auc',n_repeats=5,random_state=seed,n_jobs=1)
        pd.DataFrame({'feature':FEATURES,'mean_auc_drop':imp.importances_mean,'sd_auc_drop':imp.importances_std}).to_csv(base/'heldout_permutation.csv',index=False)
        for quality in CONFIG['jpeg']:
            z=score(est,FQ[quality][te]); q=export_predictions(te,expit(z),calibrated(sig,z),dict(tag,model=name,jpeg_quality=quality))
            q.to_csv(base/f'jpeg_{quality}.csv',index=False)
    write_json(base/'done.json',{'model':name,'tag':tag,'calibration':'Fit only on held-out training groups','signature':CONFIG['signature']})
    return pred
def save_partition(base,core,val,cal,te):
    pieces=[]
    for role,idx in [('train',core),('early_stop',val),('calibration',cal),('test',te)]:
        d=DATA.iloc[np.asarray(idx,dtype=int)][['image_id','group']].copy(); d['role']=role; pieces.append(d)
    joined=pd.concat(pieces)
    assert joined.groupby('group').role.nunique().max()==1
    joined.to_csv(base/'split.csv',index=False)

def m5_dataset(ids,quality=None):
    import torch
    from torchvision.models import EfficientNet_B0_Weights
    transform=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    class Images(torch.utils.data.Dataset):
        def __len__(self): return len(ids)
        def __getitem__(self,i):
            row=DATA.iloc[ids[i]]
            return transform(load_image(row.path,quality)),torch.tensor(float(row.label),dtype=torch.float32)
    return Images()
def m5_model():
    import torch
    from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
    model=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for par in model.parameters(): par.requires_grad=False
    for block in list(model.features.children())[-2:]:
        for par in block.parameters(): par.requires_grad=True
    model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,1)
    return model.cuda()
def loader(ids,shuffle=False,seed=42,quality=None):
    import torch
    return torch.utils.data.DataLoader(m5_dataset(ids,quality),batch_size=CONFIG['batch_size'],shuffle=shuffle,
        num_workers=2 if os.name!='nt' else 0,generator=torch.Generator().manual_seed(seed),pin_memory=True)
def m5_logits(model,ids,quality=None):
    import torch
    model.eval(); out=[]
    with torch.inference_mode():
        for xx,yy in loader(ids,quality=quality):
            budget(); out.append(model(xx.cuda()).squeeze(1).cpu().numpy())
    return np.concatenate(out)
def m5(tr,te,seed,tag):
    import torch
    budget(); key=stable(json.dumps(dict(tag,model='M5'),sort_keys=True)); base=OUT/'experiments'/key
    base.mkdir(parents=True,exist_ok=True)
    if (base/'done.json').exists(): return pd.read_csv(base/'predictions.csv')
    reset(seed); y=DATA.label.to_numpy(); g=DATA.group.to_numpy()
    core,val,cal=split_groups(y[tr],g[tr],seed); core,val,cal=tr[core],tr[val],tr[cal]
    save_partition(base,core,val,cal,te); model=m5_model()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(); best=float('inf'); stale=0; history=[]

    start=time.monotonic()
    for epoch in range(CONFIG['epochs']):
        budget(); model.train()

        for block in list(model.features.children())[:-2]: block.eval()
        losses=[]
        for xx,yy in loader(core,True,seed+epoch):
            budget(); opt.zero_grad(set_to_none=True)
            loss=criterion(model(xx.cuda()).squeeze(1),yy.cuda()); loss.backward(); opt.step(); losses.append(loss.item())
        z=m5_logits(model,val); v=float(log_loss(y[val],expit(z),labels=[0,1]))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':v,'seconds':time.monotonic()-start})
        say(f'M5 {tag} epoch {epoch+1}, validation loss {v:.4f}')
        pd.DataFrame(history).to_csv(base/'training_history.csv',index=False)
        if v<best-1e-5:
            best=v; stale=0; torch.save(model.state_dict(),base/'best_weights.pt')
        else: stale+=1
        if stale>=CONFIG['patience']: break
    model.load_state_dict(torch.load(base/'best_weights.pt',map_location='cuda',weights_only=True))
    cz=m5_logits(model,cal); temp=temperature(cz,y[cal]); tz=m5_logits(model,te)
    cp=expit(cz/temp); p=expit(tz/temp)
    pred=export_predictions(te,expit(tz),p,dict(tag,model='M5')); pred.to_csv(base/'predictions.csv',index=False)
    pd.DataFrame({'row_id':cal,'label':y[cal],'raw_logit':cz,'probability':cp}).to_csv(base/'calibration.csv',index=False)
    rows=operational_rows(y[cal],cp,y[te],p,dict(tag,model='M5'))
    if tag['evaluation']=='main': rows=operating_uncertainty(rows,y[te],p,g[te])
    pd.DataFrame(rows).to_csv(base/'operational.csv',index=False)
    write_json(base/'calibration.json',{'temperature':temp,'early_stop_best_loss':best,'training_seconds':time.monotonic()-start})
    if tag['evaluation']=='main':
        for quality in CONFIG['jpeg']:
            qz=m5_logits(model,te,quality); export_predictions(te,expit(qz),expit(qz/temp),dict(tag,model='M5',jpeg_quality=quality)).to_csv(base/f'jpeg_{quality}.csv',index=False)
        if seed==SEEDS[0]: gradcam(model,te,p,base)
    write_json(base/'done.json',{'model':'M5','tag':tag,'signature':CONFIG['signature'],'metrics_probabilities':'cross-fitted calibrated probabilities'})
    del model,opt; torch.cuda.empty_cache(); return pred

def gradcam(model,ids,p,base):
    import torch
    y=DATA.label.to_numpy()[ids]; choices=[]
    for label in [0,1]:
        for predicted in [0,1]:
            cand=np.flatnonzero((y==label)&((p>=.5)==predicted))
            if len(cand): choices.append(int(cand[np.argmin(abs(p[cand]-np.median(p[cand])))]))
    selected=[]
    for k in choices:
        row=DATA.iloc[ids[k]]; im=load_image(row.path)
        xx,_=m5_dataset([ids[k]])[0]; activ=[]
        def hook(module,args,out):
            out.retain_grad(); activ.append(out)
        handle=model.features[-1].register_forward_hook(hook)
        model.zero_grad(set_to_none=True); z=model(xx.unsqueeze(0).cuda())[0,0]; z.backward(); handle.remove()
        a=activ[0]; heat=(a.grad.mean(dim=(2,3),keepdim=True)*a).sum(1).relu().detach().cpu().numpy()[0]
        heat=heat/(heat.max()+1e-8)
        np.save(base/f'gradcam_{row.image_id}.npy',heat)
        selected.append({'image_id':row.image_id,'label':int(row.label),'probability':float(p[k]),'selection':'Nearest group-cell median among held-out real/fake x predicted real/fake images',
            'interpretation':'Qualitative attribution; not a validated damage localization or causal explanation'})
    write_json(base/'gradcam_cases.json',selected)

def paired_features(all_df,all_F):
    idx={aid:i for i,aid in enumerate(all_df.image_id)}; bypath={r.relative_path:i for i,r in all_df.iterrows()}; rows=[]
    for gen in sorted(all_df.loc[all_df.bucket=='DeepFake','generator'].unique()):
        edits=all_df[all_df.generator==gen]; dif=[]; grp=[]
        for i,r in edits.iterrows():
            j=bypath.get(r.source_relative_path)
            if j is not None: dif.append(all_F[i]-all_F[j]); grp.append(r.group)
        if not dif: continue
        d=pd.DataFrame(dif,columns=FEATURES); d['source_review']=grp

        d=d.groupby('source_review')[FEATURES].mean()
        for f in FEATURES:
            a=d[f].to_numpy(); a=a[np.isfinite(a)]; sd=float(a.std(ddof=1)) if len(a)>1 else 0
            pp=float(ttest_1samp(a,0).pvalue) if len(a)>1 and sd>0 else 1.
            rng=np.random.default_rng(42); bs=np.array([rng.choice(a,len(a)).mean() for _ in range(500)]) if len(a) else np.array([np.nan])
            rows.append({'generator':gen,'feature':f,'artifact_pairs':len(dif),'independent_review_pairs':len(a),
                'mean_edit_minus_source':float(a.mean()),'paired_dz':float(a.mean()/sd) if sd else np.nan,
                'mean_ci_lo':float(np.quantile(bs,.025)),'mean_ci_hi':float(np.quantile(bs,.975)),'p':pp,
                'interpretation':'Synthetic edit versus its undamaged source; not genuine damage versus forgery'})
    out=pd.DataFrame(rows)
    if len(out): out['p_holm_all_features_generators']=holm(out.p)
    out.to_csv(OUT/'paired_artifact_effects.csv',index=False)

def plan_experiments():
    y=DATA.label.to_numpy(); g=DATA.group.to_numpy(); tasks=[]; unavailable=[]; transfer=outer(y,g,42)
    names=['M1','M1GB','B1','B2','B3','E0','C0','M3','M4','T0','M5']+(['UFD'] if UFD is not None else [])
    missing_text=bool((DATA.text.str.strip().str.len()==0).any())
    if missing_text:
        names=[n for n in names if n not in ['M4','T0']]
        unavailable.append({'evaluation':'main','reason':'Missing review text: full-cohort M4 and T0 excluded. See text_coverage.json.'})
        say('Missing review text: M4/T0 excluded from full-cohort comparison, recorded explicitly.')
    for seed in SEEDS:
        for fold,(tr,te) in enumerate(outer(y,g,seed)):
            for n in names: tasks.append((n,tr,te,seed,{'evaluation':'main','seed':seed,'fold':fold}))
    for category in sorted(DATA.category.unique()):
        te=np.flatnonzero(DATA.category.to_numpy()==category); tr=np.flatnonzero((DATA.category.to_numpy()!=category)&~np.isin(g,g[te]))
        if len(np.unique(y[te]))<2:
            unavailable.append({'evaluation':'LODO_category','held_out':category,'reason':'Test category lacks both classes'}); continue
        for n in ['M1','M5']: tasks.append((n,tr,te,42,{'evaluation':'LODO_category','seed':42,'held_out':category}))
    generators=sorted(DATA.loc[DATA.label==1,'generator'].unique())
    for held in generators:
        for fold,(train_sources,test_sources) in enumerate(transfer):
            tr=train_sources[(y[train_sources]==0)|(DATA.generator.to_numpy()[train_sources]!=held)]
            te=test_sources[(y[test_sources]==0)|(DATA.generator.to_numpy()[test_sources]==held)]
            for n in ['M1','M5']: tasks.append((n,tr,te,42,{'evaluation':'LOGO_source_disjoint','seed':42,'held_out':held,'fold':fold}))
    damages=sorted(d for d in DATA.loc[DATA.label==1,'damage'].unique() if d!='unknown')
    for damage in damages:
        for fold,(train_sources,test_sources) in enumerate(transfer):
            tr=train_sources[(y[train_sources]==0)|(DATA.damage.to_numpy()[train_sources]!=damage)]
            te=test_sources[(y[test_sources]==0)|(DATA.damage.to_numpy()[test_sources]==damage)]
            if y[te].sum()<5:
                unavailable.append({'evaluation':'LOLO_synthetic_damage','held_out':damage,'fold':fold,'reason':'Fewer than five synthetic test images'}); continue
            for n in ['M1','M5']: tasks.append((n,tr,te,42,{'evaluation':'LOLO_synthetic_damage','seed':42,'held_out':damage,'fold':fold}))
    for source in generators:
        for target in generators:
            for fold,(train_sources,test_sources) in enumerate(transfer):
                tr=train_sources[(y[train_sources]==0)|(DATA.generator.to_numpy()[train_sources]==source)]
                te=test_sources[(y[test_sources]==0)|(DATA.generator.to_numpy()[test_sources]==target)]
                tasks.append(('M1',tr,te,42,{'evaluation':'cross_generator_source_disjoint','source':source,'target':target,'seed':42,'fold':fold}))

    mapping={}; parent={str(v):str(v) for v in DATA.group.unique()}
    def find(v):
        while parent[v]!=v: parent[v]=parent[parent[v]]; v=parent[v]
        return v
    for groups in DATA.groupby('product_key').group.unique():
        for gg in groups[1:]: parent[find(str(gg))]=find(str(groups[0]))
    products=np.array([find(str(v)) for v in g])
    try:
        for fold,(tr,te) in enumerate(outer(y,products,42)):
            tasks.append(('M1',tr,te,42,{'evaluation':'product_disjoint_sensitivity','seed':42,'fold':fold}))
    except ValueError as e:
        unavailable.append({'evaluation':'product_disjoint_sensitivity','reason':str(e)}); say('Product sensitivity unavailable: '+str(e))
    write_json(OUT/'unavailable_analyses.json',unavailable)
    return tasks,names

def bootstrap_metrics(y,p,g,seed=42):
    rng=np.random.default_rng(seed); unique=np.unique(g); ix={v:np.flatnonzero(g==v) for v in unique}; samples=[]
    for _ in range(CONFIG['bootstrap']):
        ids=np.concatenate([ix[v] for v in rng.choice(unique,len(unique))])
        if len(np.unique(y[ids]))==2: samples.append(metrics(y[ids],p[ids]))
    b=pd.DataFrame(samples); result={}
    for name in ['auc','ap','brier','log_loss','ece_10','fpr','tpr']:
        result[name+'_lo']=float(b[name].quantile(.025)); result[name+'_hi']=float(b[name].quantile(.975))
    return result
def collate(status):
    preds=[]; ops=[]; jpeg=[]; imps=[]
    for base in sorted((OUT/'experiments').glob('*')):
        if not (base/'done.json').exists(): continue
        preds.append(pd.read_csv(base/'predictions.csv'))
        ops.append(pd.read_csv(base/'operational.csv'))
        for p in base.glob('jpeg_*.csv'): jpeg.append(pd.read_csv(p))
        if (base/'heldout_permutation.csv').exists():
            d=pd.read_csv(base/'heldout_permutation.csv'); d['experiment']=base.name; imps.append(d)
    if not preds: return
    allp=pd.concat(preds,ignore_index=True); allp.to_csv(OUT/'predictions_all.csv',index=False)
    pd.concat(ops,ignore_index=True).to_csv(OUT/'operating_points.csv',index=False)
    table=[]; rawtable=[]; detail=[]; main=allp[allp.evaluation=='main']
    for (model,seed),d in main.groupby(['model','seed']):
        if len(d)!=len(DATA) or d.image_id.nunique()!=len(DATA): continue
        vals=metrics(d.label.to_numpy(),d.probability.to_numpy()); ci=bootstrap_metrics(d.label.to_numpy(),d.probability.to_numpy(),d.group.to_numpy())
        table.append(dict(model=model,seed=int(seed),**vals,**ci))
        rawtable.append(dict(model=model,seed=int(seed),**metrics(d.label.to_numpy(),d.raw_probability.to_numpy())))
        for category,part in d.groupby('category'): detail.append(dict(model=model,seed=int(seed),category=category,**metrics(part.label.to_numpy(),part.probability.to_numpy())))
    pd.DataFrame(table).to_csv(OUT/'main_metrics_by_seed.csv',index=False); pd.DataFrame(rawtable).to_csv(OUT/'raw_uncalibrated_main_metrics.csv',index=False)
    dt=pd.DataFrame(detail); dt.to_csv(OUT/'category_metrics.csv',index=False)
    if len(dt): dt.groupby(['model','seed'])[['auc','ap','brier','ece_10','fpr','tpr']].mean().reset_index().to_csv(OUT/'macro_category_metrics.csv',index=False)
    transfer=[]
    for name,dd in allp[allp.evaluation!='main'].groupby('evaluation'):
        keys=['model']+(['source','target'] if 'source' in dd and dd.source.notna().any() else ['held_out'] if 'held_out' in dd and dd.held_out.notna().any() else [])
        for tags,part in dd.groupby(keys):
            if not isinstance(tags,tuple): tags=(tags,)
            transfer.append(dict(evaluation=name,**dict(zip(keys,tags)),**metrics(part.label.to_numpy(),part.probability.to_numpy())))
    pd.DataFrame(transfer).to_csv(OUT/'transfer_metrics.csv',index=False)
    if jpeg:
        jq=pd.concat(jpeg); jq.to_csv(OUT/'jpeg_predictions.csv',index=False); jt=[]
        for tags,d in jq.groupby(['model','seed','jpeg_quality']): jt.append(dict(zip(['model','seed','jpeg_quality'],tags),**metrics(d.label.to_numpy(),d.probability.to_numpy())))
        pd.DataFrame(jt).to_csv(OUT/'compression_metrics.csv',index=False)
    if imps: pd.concat(imps).to_csv(OUT/'heldout_permutation_importance.csv',index=False)
    completed=[d for _,d in main.groupby(['model','seed']) if len(d)==len(DATA) and d.image_id.nunique()==len(DATA)]
    if completed: model_comparison(pd.concat(completed))
    write_json(OUT/'results_summary.json',{'status':status,'complete_main_seed_model_rows':len(table),
      'models':sorted(main.model.unique()),'main_n':len(DATA),'calibrated_probabilities_used':True,
      'note':'Metrics and uncertainty describe curated benchmark detection, not fraud guilt, operational prevalence, or case admissibility.'})

def model_comparison(main):

    d=main.copy(); d['loss']=-(d.label*np.log(np.clip(d.probability,1e-7,1-1e-7))+(1-d.label)*np.log(np.clip(1-d.probability,1e-7,1-1e-7)))
    wide=d.groupby(['group','model']).loss.mean().unstack(); rows=[]
    if 'M1' not in wide: return
    for name in wide.columns:
        if name=='M1': continue
        delta=(wide[name]-wide.M1).dropna().to_numpy(); sd=delta.std(ddof=1)
        rows.append({'model':name,'reference':'M1','source_groups':len(delta),'mean_logloss_difference':delta.mean(),
          'paired_dz':delta.mean()/sd if sd else np.nan,'p':ttest_1samp(delta,0).pvalue if sd else 1.,
          'negative_means':'Lower loss than M1; exploratory comparison, not independent replications'})
    t=pd.DataFrame(rows)
    if len(t): t['p_holm']=holm(t.p)
    t.to_csv(OUT/'paired_model_comparisons.csv',index=False)

def package(status,failed=None):
    try: collate(status)
    except Exception: say('Collation error: '+traceback.format_exc()); status='partial_collation_failed'
    write_json(OUT/'run_status.json',{'status':status,'failure':failed,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'signature':CONFIG.get('signature')})
    write_json(OUT/'checksums.json',{p.relative_to(OUT).as_posix():hash_file(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='checksums.json' and p.suffix!='.tmp'})
    sync()
    target=ROOT/'RefundProof_FraudBench_OUTPUT.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.tmp': z.write(p,'RefundProof_FraudBench/'+p.relative_to(OUT).as_posix())
    if API is not None and REPO:
        API.upload_file(path_or_fileobj=str(target),path_in_repo=target.name,repo_id=REPO,repo_type='dataset',commit_message='Save downloadable results ZIP')
        say('Output ZIP: https://huggingface.co/datasets/'+REPO+'/tree/main')
    if PERSIST is not None:
        shutil.copy2(target,PERSIST/target.name)
        say('Output ZIP saved to Google Drive: '+str(PERSIST/target.name))

def main():
    global API,DATA,FQ,UFD
    import torch
    from huggingface_hub import HfApi,snapshot_download
    if not torch.cuda.is_available(): raise RuntimeError('A dedicated CUDA GPU is required. The launcher itself can use CPU.')
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'experiments').mkdir(exist_ok=True)
    (OUT/'source').mkdir(exist_ok=True)
    for name in ['runner.py','legacy_features.py','requirements.txt','SCIENTIFIC_PROTOCOL.md','VALIDATION.json']:
        p=Path(__file__).parent/name
        if p.exists(): shutil.copy2(p,OUT/'source'/name)
    API=HfApi(token=os.environ.get('HF_TOKEN'))
    if PERSIST is not None:
        PERSIST.mkdir(parents=True,exist_ok=True)
        if (PERSIST/'results').exists(): shutil.copytree(PERSIST/'results',OUT,dirs_exist_ok=True)
    else:
        if not REPO: raise ValueError('Use Drive persistence or a private output repository')
        info=API.repo_info(REPO,repo_type='dataset')
        if not info.private: raise ValueError('Results repository must be private')

    if REPO:
        try: snapshot_download(REPO,repo_type='dataset',local_dir=str(ROOT/'restore'),allow_patterns=['results/*'])
        except Exception as e: say('No prior checkpoints restored: '+str(type(e).__name__))
    if (ROOT/'restore/results').exists(): shutil.copytree(ROOT/'restore/results',OUT,dirs_exist_ok=True)
    old=json.loads((OUT/'configuration.json').read_text()) if (OUT/'configuration.json').exists() else {}
    rev=old.get('dataset_revision') or API.dataset_info(CONFIG['dataset']).sha
    CONFIG['dataset_revision']=rev; CONFIG['clip_revision']=old.get('clip_revision'); CONFIG['ufd_clip_revision']=old.get('ufd_clip_revision')
    CONFIG['runner_sha256']=hash_file(__file__)
    CONFIG['signature']=stable(json.dumps({k:v for k,v in CONFIG.items() if k not in ['signature','clip_revision','ufd_clip_revision']},sort_keys=True))
    if old and old.get('signature')!=CONFIG['signature']: raise ValueError('Existing results were made with a different protocol. Use a new output repository.')
    write_json(OUT/'configuration.json',CONFIG)
    write_json(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
      'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
      'source_sha256':hash_file(__file__),'installed_packages':{d.metadata['Name']:d.version for d in __import__('importlib.metadata',fromlist=['distributions']).distributions()}})
    say('Downloading pinned FraudBench snapshot; no image-generation stage will run.')
    root=Path(snapshot_download(CONFIG['dataset'],repo_type='dataset',revision=rev,local_dir=str(ROOT/'dataset'),max_workers=4))
    all_df,DATA=audit_dataset(root); all_F=get_forensic(all_df); F=all_F[all_df.eligible.to_numpy()]
    paired_features(all_df,all_F); sync(); emb=embeddings(DATA)
    baseline_failure=None
    try: UFD=ufd_logits(DATA)
    except BudgetStop: raise
    except Exception as e:
        UFD=None; baseline_failure=str(e); write_json(OUT/'ufd_failure.json',{'error':str(e),'traceback':traceback.format_exc()}); say('Pretrained UFD unavailable; continuing with explicitly recorded missing comparison: '+str(e))
    FQ={q:get_forensic(DATA,q) for q in CONFIG['jpeg']}
    tasks,names=plan_experiments(); write_json(OUT/'experiment_plan.json',{'total':len(tasks),
      'counts':pd.Series([t[4]['evaluation']+' / '+t[0] for t in tasks]).value_counts().to_dict(),
      'models':names,'note':'All planned analyses run automatically; invalid splits and budget exhaustion are recorded, not treated as complete.'})
    inputs={name:model_input(name,F,emb['efn'],emb['clip'],DATA.text.to_numpy()) for name in names if name!='M5'}
    failures=[]; main_time=[]
    sync()
    for k,(name,tr,te,seed,tag) in enumerate(tasks):
        budget(); say(f'Experiment {k+1}/{len(tasks)}: {name} {tag}'); start=time.monotonic()
        assert not set(DATA.group.iloc[tr])&set(DATA.group.iloc[te])
        try:
            if name=='M5': m5(tr,te,seed,tag)
            else: shallow(name,inputs[name],tr,te,seed,tag)
        except BudgetStop: raise
        except Exception as e:
            failures.append({'model':name,'tag':tag,'error':str(e),'traceback':traceback.format_exc()})
            write_json(OUT/'failed_experiments.json',failures); say('Experiment failed; saved diagnosis: '+str(e))

            if tag['evaluation']=='main': raise RuntimeError('Primary experiment failed; inspect failed_experiments.json') from e
        sync()
        if name=='M5' and tag['evaluation']=='main':
            elapsed=time.monotonic()-start
            if elapsed>30: main_time.append(elapsed)
            if main_time:
                remaining=sum(t[0]=='M5' for t in tasks[k+1:])*np.mean(main_time)/3600
                say(f'Measured rough remaining M5 time: {remaining:.1f} hours; category sizes and early stopping may change this.')
    package('partial_failed_baseline' if baseline_failure else 'complete' if not failures else 'partial_failed_experiments')

if __name__=='__main__':
    try: main()
    except BudgetStop as e:
        say(str(e)); package('partial_time_limit',str(e)); sys.exit(2)
    except Exception as e:
        say(traceback.format_exc())
        if OUT.exists():
            try: package('failed',str(e))
            except Exception: print('Final upload failed. Read the HF Job logs; last successful checkpoint upload is retained.',flush=True)
        raise
