

RUN_NAME = "RefundProof_FORGELENS_NEXT"
SESSION_HOURS = 8
RUN_EXTERNAL = True
RUN_COMPRESSION = True
BATCH_SIZE = 4
BOOTSTRAP_REPEATS = 500

import os,sys,time,json,hashlib,zipfile,shutil,threading,traceback,urllib.request,io,gc,random
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,pandas as pd,torch
from PIL import Image,ImageOps
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,roc_curve
import joblib

VERSION='RefundProof-ForgeLens-next-1.0'
REPO_COMMIT='9660c91d0c40dcdf3bef5155bd1c38ec866db13e'
REPO_ARCHIVE_SHA='c73c0b1958ded88a9a983c9cb4cf152275eebabf5f509d8d3c81859fe04a8276'
MODELS={
 'ForgeLens_GenImage':{'id':'1ZGmFPDJFtiJBxe3_mQGP_ZKPO_XXzjWc','bytes':1816067013,'wsgm':4,
  'sha256':'05a8b0ac9a36f4ebf4ce7b0031471326463acff3955be462251f2fe13808ce8f'},
 'ForgeLens_ProGAN1600':{'id':'1JxfFqVrX50U5FFR_Wm1BGYVtIi-IX_sH','bytes':1835013669,'wsgm':12,
  'sha256':'968ee215ef4109eaa395a45612fe0273196c92f3d072d6b47b378e90ddd1766d'}}
SEEDS=[42,123,777];FPR_TARGETS=[.01,.05,.10];COST_RATIOS=[2,5,10,20];PREVALENCES=[.01,.05,.10,.25,.50]
ROOT=Path('/kaggle/working')/RUN_NAME;OUT=ROOT/'results';INPUTS=Path('/kaggle/input')
CACHE=Path('/tmp/refundproof_forgelens_cache');STAGE='startup';SIGNATURE='';DEADLINE=0;HF_TOKEN=''
ZIP_NAME='RefundProof_FORGELENS_NEXT_OUTPUT.zip'

class BudgetStop(Exception):pass
def say(message):
    global STAGE
    STAGE=str(message);line=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')+' '+STAGE
    print(line,flush=True)
    if OUT.exists():
        with (OUT/'run.log').open('a',encoding='utf-8') as stream:stream.write(line+'\n')
def check():
    if time.monotonic()>DEADLINE:raise BudgetStop('Session allowance reached; this ZIP can resume the same cell.')
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8');temporary.replace(path)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def csv(path):return pd.read_csv(path,float_precision='round_trip')
def safe_extract(archive,destination,prefix='',allow=None):
    destination=Path(destination).resolve()
    for entry in archive.infolist():
        if entry.is_dir() or not entry.filename.startswith(prefix):continue
        relative=entry.filename[len(prefix):]
        if allow is not None and not allow(relative):continue
        target=(destination/relative).resolve()
        if not target.is_relative_to(destination):raise ValueError('Unsafe archive path')
        target.parent.mkdir(parents=True,exist_ok=True)
        with archive.open(entry) as source,target.open('wb') as result:shutil.copyfileobj(source,result)
def pack(status,failure=None):
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'run_status.json',{'status':status,'stage':STAGE,'failure':failure,'signature':SIGNATURE,
        'version':VERSION,'finished_utc':datetime.now(timezone.utc).isoformat(),'human_verified':False,
        'scope':'Repeated benchmark challenge; no independently confirmed fraud cases or automatic refund decision claims'})
    if status=='computational_complete_annotation_pending':
        write_json(OUT/'checksums.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json' and p.suffix!='.tmp'})
    target=ROOT/ZIP_NAME;temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix!='.tmp':archive.write(path,'RefundProof_FORGELENS_NEXT/'+path.relative_to(OUT).as_posix())
    temporary.replace(target)
    print('Saved output checkpoint:',target,flush=True)

def input_files(filename):

    visited=set();found={}
    for directory,subdirs,files in os.walk(INPUTS,followlinks=True):
        resolved=Path(directory).resolve()
        if resolved in visited:
            subdirs[:]=[];continue
        visited.add(resolved)
        if filename in files:
            candidate=Path(directory)/filename
            found.setdefault(candidate.resolve(),candidate)
    return sorted(found.values(),key=str)

def restore_results():
    if (OUT/'protocol.json').exists():return
    matches=input_files(ZIP_NAME)
    if len(matches)>1:raise ValueError('Attach one comparison checkpoint ZIP.')
    if matches:
        with zipfile.ZipFile(matches[0]) as archive:safe_extract(archive,OUT,'RefundProof_FORGELENS_NEXT/')

def input_manifest():
    matches=input_files('RefundProof_Study_Inputs.zip')
    if not matches:matches=input_files('RefundProof_RECOVERED_COMPLETE_OUTPUT.zip')
    if not matches:matches=input_files('RefundProof_VALIDATOR_REPAIR_OUTPUT.zip')
    manifests=input_files('primary_manifest.csv')
    folders=[p.parent for p in manifests if (p.parent/'media').is_dir() and (p.parent/'study_configuration.json').is_file()]
    say('Detected study folders: '+str([str(p) for p in folders]))
    if len(matches)>1:raise ValueError('Attach one complete study input ZIP.')
    source=CACHE/'study_input';source.mkdir(parents=True,exist_ok=True)
    keep=lambda r:r.startswith(('media/','changed_masks/','external/external_manifest.csv','external/manifest_all.csv','external/external_provenance.json','external/final_probabilities_')) or ('/' not in r and r.endswith(('.csv','.json')))
    if matches:
        say('Restoring saved images and manifests; no generation.')
        with zipfile.ZipFile(matches[0]) as archive:safe_extract(archive,source,'RefundProof_PAPER/',keep)
    elif len(folders)==1:source=folders[0]
    else:raise ValueError('Could not identify exactly one complete study folder. Found manifests: '+str([str(p) for p in manifests])+'. Each needs adjacent media/ and study_configuration.json. Detected folders: '+str([str(p) for p in folders]))
    config=json.loads((source/'study_configuration.json').read_text())
    if config.get('signature')!='bf975e1b41e33cf57b4f9933':raise ValueError('The input signature must match the retained study cohort.')
    primary=csv(source/'primary_manifest.csv');all_images=csv(source/'manifest_all.csv')
    if len(primary)!=462 or primary.image_id.nunique()!=462:raise ValueError('Unexpected retained main cohort.')
    metadata=json.loads((source/'external/external_provenance.json').read_text())
    external=csv(source/'external/external_manifest.csv')
    if len(external)!=5590 or external.image_id.nunique()!=5590:raise ValueError('Unexpected frozen external cohort.')
    for frame in [primary,all_images]:
        frame['local_path']=frame.relative_path.map(lambda value:str(source/value))
        for row in frame.itertuples():
            if not Path(row.local_path).is_file() or sha(row.local_path)!=row.sha256:raise ValueError('Saved image missing or differs: '+row.image_id)
    benign=all_images[all_images.variant=='benign'].copy()
    if len(benign)!=68:raise ValueError('Unexpected benign-control cohort.')
    raw_lookup=csv(source/'external/manifest_all.csv').set_index('image_id')
    external['raw_sha256']=raw_lookup.loc[external.image_id,'sha256'].to_numpy()
    external['hf_relative']=external.original_external_path.map(lambda value:str(value).split('/tmp/refundproof_fraudbench/',1)[1])
    return source,primary.reset_index(drop=True),benign.reset_index(drop=True),external.reset_index(drop=True),metadata

def download_file(identifier,path,size,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==size and sha(path)==expected:return path
    partial=path.with_suffix('.part')
    for attempt in range(5):
        check();offset=partial.stat().st_size if partial.exists() else 0
        if offset==size:
            partial.replace(path)
            if sha(path)==expected:return path
            path.unlink();offset=0
        url='https://drive.usercontent.google.com/download?id='+identifier+'&export=download&confirm=t&rp_offset='+str(offset)
        headers={'Range':'bytes='+str(offset)+'-'} if offset else {}
        try:
            say('Downloading author checkpoint '+path.name+f' from byte {offset:,}; attempt {attempt+1}/5')
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
                if 'application/octet-stream' not in response.headers.get('Content-Type',''):raise RuntimeError('Google Drive returned HTML instead of weights; attach the author checkpoint manually.')
                if offset and response.status!=206:offset=0
                if offset and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):
                    raise RuntimeError('Unexpected resumed-download byte range')
                mode='ab' if offset else 'wb';last=offset
                with partial.open(mode) as stream:
                    while True:
                        check();chunk=response.read(4*1024*1024)
                        if not chunk:break
                        stream.write(chunk);offset+=len(chunk)
                        if offset-last>=128*1024*1024:say(f'Checkpoint {path.name}: {offset/size:.1%}');last=offset
            if partial.stat().st_size!=size:raise RuntimeError('Checkpoint download is incomplete')
            partial.replace(path)
            if sha(path)!=expected:path.unlink();raise RuntimeError('Checkpoint SHA256 differs from the author file audited locally')
            return path
        except BudgetStop:raise
        except Exception as error:
            say('Checkpoint retry: '+str(error))
            if attempt==4:raise
            time.sleep(min(30,5*(attempt+1)))

def official_source():
    path=CACHE/'ForgeLens-source.zip';directory=CACHE/'ForgeLens';directory.mkdir(parents=True,exist_ok=True)
    if not path.exists() or sha(path)!=REPO_ARCHIVE_SHA:
        say('Downloading pinned official ForgeLens implementation.')
        with urllib.request.urlopen('https://codeload.github.com/Yingjian-Chen/ForgeLens/zip/'+REPO_COMMIT,timeout=120) as response,path.open('wb') as stream:shutil.copyfileobj(response,stream)
    if sha(path)!=REPO_ARCHIVE_SHA:raise RuntimeError('Official repository archive hash changed.')
    with zipfile.ZipFile(path) as archive:safe_extract(archive,directory,'ForgeLens-'+REPO_COMMIT+'/',lambda r:'__pycache__' not in r)
    sys.path.insert(0,str(directory));return directory

def load_detector(path,settings,repo,device):

    state=torch.load(str(path),map_location='cpu',weights_only=True,mmap=True)['model_state_dict']
    arguments=sys.argv;sys.argv=['forgelens_inference','--WSGM_count',str(settings['wsgm']),'--WSGM_reduction_factor','4']
    try:
        from models.network.clip import clip as clip_module
        from models.network.clip import model as clip_model
        from models.network.net_stage2 import net_stage2
    finally:sys.argv=arguments
    if not Path(clip_model.__file__).resolve().is_relative_to(repo.resolve()):raise RuntimeError('Conflicting models module in this session; use a fresh Kaggle notebook.')
    clip_model.opt.WSGM_count=settings['wsgm'];clip_model.opt.WSGM_reduction_factor=4
    prefix='backbone.backbone.'
    backbone_state={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    original_loader=clip_module.load
    clip_module.load=lambda *args,**kwargs:(clip_model.build_model(dict(backbone_state)).float(),None)
    try:model=net_stage2(SimpleNamespace(FAFormer_layers=2,FAFormer_head=2,FAFormer_reduction_factor=1),train=False)
    finally:clip_module.load=original_loader
    model=model.float();model.load_state_dict(state,strict=True);model.eval().to(device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    del state,backbone_state;gc.collect()
    return model

MEAN=np.array([.48145466,.4578275,.40821073],dtype=np.float32)[:,None,None]
STD=np.array([.26862954,.26130258,.27577711],dtype=np.float32)[:,None,None]
def rgb(path):
    with Image.open(path) as image:return ImageOps.exif_transpose(image).convert('RGB')
def square(image):return ImageOps.pad(image,(512,512),method=Image.Resampling.LANCZOS,color=(127,127,127),centering=(.5,.5))
def native_crop(image):

    width,height=image.size
    if min(width,height)<224:
        w=width*int(np.ceil(224/width));h=height*int(np.ceil(224/height));expanded=Image.new('RGB',(w,h))
        for x in range(0,w,width):
            for y in range(0,h,height):expanded.paste(image,(x,y))
        image=expanded;width,height=image.size
    x=int(round((width-224)/2));y=int(round((height-224)/2));return image.crop((x,y,x+224,y+224))
def tensor(image):
    values=np.asarray(image,dtype=np.float32).transpose(2,0,1)/255
    return torch.from_numpy(((values-MEAN)/STD).copy())
def image_views(image,external=False,quality=None):
    normalized=square(image)
    if quality is not None:
        stream=io.BytesIO();normalized.save(stream,format='JPEG',quality=int(quality));stream.seek(0);normalized=Image.open(stream).convert('RGB')
    views=[normalized.resize((224,224),Image.Resampling.BICUBIC)]
    views += [normalized.crop((x,y,x+224,y+224)) for y in [0,144,288] for x in [0,144,288]]
    if external:views.append(native_crop(image))
    return views
def infer_views(model,views,device):
    outputs=[]
    with torch.inference_mode():
        for start in range(0,len(views),BATCH_SIZE):
            check();batch=torch.stack([tensor(image) for image in views[start:start+BATCH_SIZE]]).to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):z=model(batch)
            z=z.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
            if not np.isfinite(z).all():raise RuntimeError('Nonfinite detector output; no fabricated score will be substituted.')
            outputs.extend(z.tolist())
    return np.asarray(outputs,dtype=np.float64)
def cache_path(name,kind,quality=None):return OUT/'features'/f'{name}_{kind}_{quality or "original"}.csv'
def cached_features(name,kind,frame,quality=None):
    path=cache_path(name,kind,quality)
    if not path.exists():return None
    cached=csv(path)
    if cached.image_id.duplicated().any():raise ValueError('Duplicate cache image IDs')
    if set(cached.image_id)!=set(frame.image_id):return None
    return cached.set_index('image_id').loc[frame.image_id].reset_index()
def extract_features(model,name,kind,frame,device,quality=None):
    path=cache_path(name,kind,quality);path.parent.mkdir(parents=True,exist_ok=True)
    cached=csv(path) if path.exists() else pd.DataFrame();records=cached.to_dict('records')
    existing=set(cached.image_id) if len(cached) else set();pending=frame[~frame.image_id.isin(existing)]
    if len(existing-set(frame.image_id)):raise ValueError('Feature cache does not match the declared dataset')
    for count,row in enumerate(pending.itertuples(),1):
        check();image=rgb(row.local_path);values=infer_views(model,image_views(image,kind=='external',quality),device)
        record={'image_id':row.image_id,'global_logit':values[0],**{f'patch_{i}':values[i+1] for i in range(9)}}
        record['official_canonical_logit']=values[5]
        if kind=='external':record['official_native_logit']=values[-1]
        records.append(record)
        if count%20==0 or count==len(pending):
            pd.DataFrame(records).to_csv(path,index=False,float_format='%.17g')
            say(f'{name}: {kind}, JPEG={quality}, {len(records)}/{len(frame)} images completed')
        if count%200==0:pack('partial_feature_extraction')
    return cached_features(name,kind,frame,quality)

def external_pixels(frame,metadata):
    from huggingface_hub import snapshot_download
    matches=[]
    first=frame.hf_relative.iloc[0]
    for candidate in input_files(Path(first).name):
        suffix=Path(first).parts
        if tuple(candidate.parts[-len(suffix):])==suffix:matches.append(candidate.parents[len(suffix)-1])
    matches=sorted(set(matches))
    if len(matches)>1:raise ValueError('Multiple matching FraudBench copies; attach only one.')
    if matches:folder=matches[0];say('Using attached FraudBench image folder.')
    else:
        if not HF_TOKEN:raise ValueError('Enable the HF_TOKEN Kaggle secret, or attach the already downloaded FraudBench dataset.')
        folder=CACHE/'FraudBench';say('Downloading the SAME 5,590 FraudBench images at the saved revision. Download progress follows.')
        for attempt in range(5):
            check()
            try:
                snapshot_download(repo_id='TristanYan/FraudBench',repo_type='dataset',revision=metadata['revision'],
                    allow_patterns=frame.hf_relative.tolist(),local_dir=str(folder),token=HF_TOKEN,max_workers=2)
                break
            except Exception as error:
                if attempt==4:raise
                say('Dataset download retry: '+str(error));time.sleep(30)
    frame=frame.copy();frame['local_path']=frame.hf_relative.map(lambda value:str(folder/value))
    for row in frame.itertuples():
        check()
        if not Path(row.local_path).is_file() or sha(row.local_path)!=row.raw_sha256:raise ValueError('FraudBench bytes differ or are missing: '+row.hf_relative)
    say('All external image byte identities verified.');return frame

def design(features,kind):
    global_score=features.global_logit.to_numpy(dtype=np.float64)[:,None]
    if kind=='GlobalCal':return global_score
    patches=features[[f'patch_{i}' for i in range(9)]].to_numpy(dtype=np.float64)
    return np.column_stack([global_score,patches.mean(1),patches.std(1),np.sort(patches,axis=1)[:,-2:].mean(1),patches.min(1)])
def split_inner(labels,groups,seed):
    for trial in range(30):
        train,cal=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+trial).split(np.zeros(len(labels)),labels,groups))
        if len(np.unique(labels[train]))==2 and len(np.unique(labels[cal]))==2:return train,cal
    raise ValueError('Cannot form two-class group-disjoint train/calibration partitions')
def fit_head(x,y,groups,seed):
    train,cal=split_inner(y,groups,seed)
    assert not set(groups[train])&set(groups[cal])
    estimator=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=seed))
    estimator.fit(x[train],y[train]);z=estimator.decision_function(x[cal])
    calibrator=LogisticRegression(C=1,max_iter=2000,random_state=seed).fit(z[:,None],y[cal])
    return estimator,calibrator,train,cal
def probability(estimator,calibrator,x):return calibrator.predict_proba(estimator.decision_function(x)[:,None])[:,1]
def rates(y,p,threshold):
    prediction=p>=threshold
    return {'fpr':float(prediction[y==0].mean()),'tpr':float(prediction[y==1].mean()),
        'real_n':int((y==0).sum()),'fake_n':int((y==1).sum()),
        'false_positives':int(prediction[y==0].sum()),'false_negatives':int((~prediction[y==1]).sum())}
def operating(cal_y,cal_p,test_y,test_p,tag):
    tag=dict(tag,probability_scope='One frozen development-fold head; not the final ensemble')
    f,t,h=roc_curve(cal_y,cal_p,drop_intermediate=False);rows=[]
    for target in FPR_TARGETS:
        eligible=np.flatnonzero(f<=target);j=eligible[np.argmax(t[eligible])]
        assert rates(cal_y,cal_p,h[j])['fpr']<=target
        rows.append(dict(tag,policy='calibration_FPR_target',target=target,threshold=float(h[j]),selection_fpr=float(f[j]),selection_tpr=float(t[j]),**rates(test_y,test_p,h[j])))
    for ratio in COST_RATIOS:
        for prevalence in PREVALENCES:
            j=int(np.argmin(ratio*(1-prevalence)*f+prevalence*(1-t)))
            rr=rates(test_y,test_p,h[j])
            rows.append(dict(tag,policy='minimum_calibration_expected_cost',cost_fp_to_fn=ratio,prevalence=prevalence,threshold=float(h[j]),
                expected_calibration_cost=float(ratio*(1-prevalence)*f[j]+prevalence*(1-t[j])),
                expected_test_cost=float(ratio*(1-prevalence)*rr['fpr']+prevalence*(1-rr['tpr'])),**rr))
    return rows
def metrics(y,p,groups,categories,intervals=True):
    y=np.asarray(y);p=np.asarray(p,dtype=np.float64);categories=np.asarray(categories);groups=np.asarray(groups)
    r=rates(y,p,.5);ece=0.
    for i in range(10):
        selected=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
        if selected.any():ece+=selected.mean()*abs(y[selected].mean()-p[selected].mean())
    category_balanced=[]
    for category in np.unique(categories):
        selected=categories==category
        if len(np.unique(y[selected]))==2:
            rr=rates(y[selected],p[selected],.5);category_balanced.append((1-rr['fpr']+rr['tpr'])/2)
    result={'n':len(y),'real':int((y==0).sum()),'fake':int(y.sum()),'auc':float(roc_auc_score(y,p)),
        'ap':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),
        'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),'ece_10':float(ece),
        'balanced_accuracy':float((1-r['fpr']+r['tpr'])/2),'macro_category_balanced_accuracy':float(np.mean(category_balanced)),**r}
    if intervals:
        rng=np.random.default_rng(42);unique=np.unique(groups);lookup={g:np.flatnonzero(groups==g) for g in unique};values=[]
        for _ in range(BOOTSTRAP_REPEATS):
            check();ids=np.concatenate([lookup[g] for g in rng.choice(unique,len(unique))])
            if len(np.unique(y[ids]))==2:values.append(roc_auc_score(y[ids],p[ids]))
        result.update(auc_lo=float(np.quantile(values,.025)),auc_hi=float(np.quantile(values,.975)))
    return result

def adaptation(name,kind,primary,benign,external,features,bfeatures,efeatures,jpeg_features):
    x=design(features,kind);y=primary.label.to_numpy();g=primary.group.to_numpy();bx=design(bfeatures,kind)
    ex=design(efeatures,kind) if RUN_EXTERNAL else None;model_name=name+'_'+kind;folder=OUT/'folds'/model_name;folder.mkdir(parents=True,exist_ok=True)
    predictions=[];external_predictions=[];benign_predictions=[];jpeg_predictions=[];operation_rows=[]
    for seed in SEEDS:
        for fold,(tr,te) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(x,y,g)):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            check();base=folder/f'{seed}_{fold}';base.mkdir(exist_ok=True);tag={'model':model_name,'seed':seed,'fold':fold}
            estimator,calibrator,a,b=fit_head(x[tr],y[tr],g[tr],seed+fold);train=tr[a];cal=tr[b]
            assert not set(g[train])&set(g[cal]) and not set(g[tr])&set(g[te])
            joblib.dump((estimator,calibrator),base/'head.joblib')
            pd.concat([primary.iloc[indices][['image_id','group']].assign(role=role) for indices,role in [(train,'train'),(cal,'calibration'),(te,'test')]]).to_csv(base/'split.csv',index=False)
            cp=probability(estimator,calibrator,x[cal]);p=probability(estimator,calibrator,x[te])
            pd.DataFrame({'image_id':primary.iloc[cal].image_id,'label':y[cal],'probability':cp}).to_csv(base/'calibration.csv',index=False,float_format='%.17g')
            frame=primary.iloc[te][['image_id','artifact_id','group','category','generator','label']].copy();frame['probability']=p
            frame['seed']=seed;frame['fold']=fold;predictions.append(frame);frame.to_csv(base/'predictions.csv',index=False,float_format='%.17g')
            operation_rows+=operating(y[cal],cp,y[te],p,dict(tag,evaluation='main'))
            if RUN_EXTERNAL:
                ep=probability(estimator,calibrator,ex);external_predictions.append(ep)
                operation_rows+=operating(y[cal],cp,external.label.to_numpy(),ep,dict(tag,evaluation='external_repeated_challenge'))
            chosen=benign.group.isin(g[te]);bp=probability(estimator,calibrator,bx[chosen]) if chosen.any() else np.array([])
            benign_predictions.append(pd.DataFrame({'image_id':benign.loc[chosen,'image_id'],'probability':bp,'seed':seed,'fold':fold}))
            for quality,jf in jpeg_features.items():
                qp=probability(estimator,calibrator,design(jf,kind)[te]);jpeg_predictions.append(pd.DataFrame({'image_id':primary.iloc[te].image_id,'probability':qp,'seed':seed,'quality':quality}))
            say(f'Grouped development head {model_name}: seed {seed}, fold {fold+1}/3 completed')
    combined=pd.concat(predictions);combined.to_csv(folder/'all_seed_predictions.csv',index=False,float_format='%.17g')
    assert combined.groupby(['image_id','seed']).size().eq(1).all() and len(combined)==len(primary)*len(SEEDS)
    final=primary[['image_id','artifact_id','group','category','generator','label']].copy()
    final['probability']=combined.groupby('image_id').probability.mean().loc[final.image_id].to_numpy()
    final.to_csv(OUT/'predictions'/f'{model_name}_main.csv',index=False,float_format='%.17g')
    rows=[dict(model=model_name,evaluation='main',**metrics(final.label,final.probability,final.group,final.category))]
    if RUN_EXTERNAL:
        ef=external[['image_id','artifact_id','group','category','generator','label']].copy();ef['probability']=np.mean(external_predictions,axis=0)
        ef.to_csv(OUT/'predictions'/f'{model_name}_external.csv',index=False,float_format='%.17g')
        rows.append(dict(model=model_name,evaluation='external_repeated_challenge',**metrics(ef.label,ef.probability,ef.group,ef.category)))
    controls=pd.concat(benign_predictions)
    assert controls.groupby(['image_id','seed']).size().eq(1).all() and len(controls)==len(benign)*len(SEEDS)
    controls.groupby('image_id').probability.mean().rename('probability').to_csv(OUT/'predictions'/f'{model_name}_benign.csv',float_format='%.17g')
    pd.DataFrame(operation_rows).to_csv(folder/'operating_points.csv',index=False,float_format='%.17g')
    if jpeg_predictions:
        jq=pd.concat(jpeg_predictions);jq.to_csv(folder/'jpeg_seed_predictions.csv',index=False,float_format='%.17g');jrows=[]
        for quality,part in jq.groupby('quality'):
            probability_vector=part.groupby('image_id').probability.mean().loc[primary.image_id].to_numpy()
            jrows.append(dict(model=model_name,quality=quality,**metrics(y,probability_vector,g,primary.category,False)))
        pd.DataFrame(jrows).to_csv(folder/'compression_metrics.csv',index=False)
    return rows

def patch_diagnostics(name,primary,features,source):
    folder=OUT/'patch_diagnostics'/name;folder.mkdir(parents=True,exist_ok=True);rows=[]
    scores=expit(features[[f'patch_{i}' for i in range(9)]].to_numpy(dtype=np.float64))
    for index,row in enumerate(primary.itertuples()):
        check();heat=np.zeros((512,512));coverage=np.zeros((512,512));k=0
        for y in [0,144,288]:
            for x in [0,144,288]:heat[y:y+224,x:x+224]+=scores[index,k];coverage[y:y+224,x:x+224]+=1;k+=1
        heat/=coverage
        if row.label==1:
            mask_path=source/'changed_masks'/Path(row.changed_mask_path).name
            mask=np.asarray(Image.open(mask_path).convert('L'))>0
            if mask.shape!=(512,512):raise ValueError('Unexpected computational mask size')
            if mask.any() and (~mask).any():
                selected=heat>=.5
                rows.append({'image_id':row.image_id,'artifact_id':row.artifact_id,'group':row.group,'generator':row.generator,
                    'change_pixel_auc':float(roc_auc_score(mask.ravel(),heat.ravel())),
                    'change_pixel_ap':float(average_precision_score(mask.ravel(),heat.ravel())),
                    'fixed_05_change_support_iou':float((selected&mask).sum()/max(1,(selected|mask).sum())),
                    'annotation_target':'Computed changed pixels; not human-verified physical damage',
                    'map_scope':'Frozen detector patch probabilities; not a causal explanation of the fitted fusion head'})
        if (index+1)%100==0:say(f'Patch diagnostic maps {name}: {index+1}/{len(primary)}')
    pd.DataFrame(rows).to_csv(folder/'change_support_metrics.csv',index=False)

def previous_baselines(source,primary,external):
    rows=[]
    for name in ['M4','M5','UFD']:
        path=source/f'final_probabilities_{name}.csv'
        if path.exists():p=csv(path).set_index('image_id').loc[primary.image_id,'probability'].to_numpy()
        else:
            previous=csv(source/'predictions_all.csv');previous=previous[(previous.evaluation=='main')&(previous.model==name)]
            if len(previous)!=len(primary)*3 or not previous.groupby(['image_id','seed']).size().eq(1).all():raise ValueError('Incomplete baseline probability records')
            p=previous.groupby('image_id').probability.mean().loc[primary.image_id].to_numpy()
        identifier='Previous_'+name
        frame=primary[['image_id','artifact_id','group','category','generator','label']].copy();frame['probability']=p
        frame.to_csv(OUT/'predictions'/f'{identifier}_main.csv',index=False,float_format='%.17g')
        rows.append(dict(model=identifier,evaluation='main',**metrics(primary.label,p,primary.group,primary.category)))
        if RUN_EXTERNAL:
            path=source/f'external/final_probabilities_{name}.csv'
            p=csv(path).set_index('image_id').loc[external.image_id,'probability'].to_numpy()
            frame=external[['image_id','artifact_id','group','category','generator','label']].copy();frame['probability']=p
            frame.to_csv(OUT/'predictions'/f'{identifier}_external.csv',index=False,float_format='%.17g')
            rows.append(dict(model=identifier,evaluation='external_repeated_challenge',**metrics(external.label,p,external.group,external.category)))
    return rows

def paired_comparisons(results):
    from scipy.stats import ttest_1samp
    rows=[]
    for evaluation,part in results.groupby('evaluation'):
        suffix='main' if evaluation=='main' else 'external'
        reference=csv(OUT/'predictions'/f'Previous_M5_{suffix}.csv').set_index('image_id')
        for model in part.model:
            if model.startswith('Previous_') or model.endswith('OfficialNative'):continue
            path=OUT/'predictions'/f'{model}_{suffix}.csv'
            if not path.exists():path=OUT/'predictions'/f'{model}_{evaluation}.csv'
            candidate=csv(path).set_index('image_id').loc[reference.index]
            assert np.array_equal(reference.label,candidate.label)
            losses=lambda frame:-(frame.label*np.log(np.clip(frame.probability,1e-7,1-1e-7))+(1-frame.label)*np.log(np.clip(1-frame.probability,1e-7,1-1e-7)))
            delta=(losses(candidate)-losses(reference)).groupby(reference.group).mean().to_numpy()
            sd=delta.std(ddof=1)
            rows.append({'evaluation':evaluation,'model':model,'reference':'Previous_M5','source_groups':len(delta),
                'mean_group_logloss_difference':float(delta.mean()),'paired_dz':float(delta.mean()/sd) if sd else None,
                'p':float(ttest_1samp(delta,0).pvalue) if sd else 1.,
                'interpretation':'Exploratory paired source-group comparison; negative means lower loss; no independent confirmation'})
    frame=pd.DataFrame(rows);p=frame.p.to_numpy();order=np.argsort(p);adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    frame['p_holm_all_declared_comparisons']=adjusted;frame.to_csv(OUT/'paired_model_comparisons.csv',index=False)

def run():
    global DEADLINE,SIGNATURE,HF_TOKEN
    if not RUN_NAME.replace('_','').replace('-','').isalnum():raise ValueError('Invalid RUN_NAME')
    if not torch.cuda.is_available():raise RuntimeError('Enable a T4 GPU. This cell performs pretrained detector inference, not image generation.')
    if tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])<(2,6):raise RuntimeError('This cell needs Torch >=2.6 for safe checkpoint loading. Use Kaggle\'s current GPU environment.')
    ROOT.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True,parents=True)
    DEADLINE=time.monotonic()+SESSION_HOURS*3600-300
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):print(datetime.now(timezone.utc).strftime('[%H:%M:%S UTC]')+' ACTIVE: '+STAGE,flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        import subprocess
        say('GPU: '+torch.cuda.get_device_name(0)+'. Installing only the small inference dependencies; preserving Kaggle Torch.')
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','ftfy==6.3.1','setuptools==80.9.0','huggingface_hub==0.36.0','PyYAML>=6.0.2'],check=True)
        try:
            from kaggle_secrets import UserSecretsClient
            HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN').strip()
        except Exception:HF_TOKEN=''
        restore_results();source,primary,benign,external,metadata=input_manifest()
        config={'version':VERSION,'models':MODELS,'repository_commit':REPO_COMMIT,'seeds':SEEDS,'fpr_targets':FPR_TARGETS,
            'cost_ratios':COST_RATIOS,'prevalences':PREVALENCES,'run_external':RUN_EXTERNAL,'run_compression':RUN_COMPRESSION,
            'bootstrap':BOOTSTRAP_REPEATS,'main_image_identity':sha(source/'primary_manifest.csv'),'external_identity':sha(source/'external/external_manifest.csv'),
            'folds':3,'inner_calibration_group_fraction':.2,'head_C':.1,'calibrator_C':1,'fusion_features':['global_logit','patch_mean','patch_std','top2_mean','patch_min'],
            'external_interpretation':'Previously inspected dataset: repeated challenge, never used for fitting or hyperparameter choice',
            'method_description':'Frozen global and overlapping-tile logits with grouped logistic score fusion and separate sigmoid calibration',
            'calibration_and_policies':'Training-source groups only; external ensemble averages frozen development-fold probabilities',
            'input_preprocessing':'Canonical 512 gray letterbox; official center crop, separate whole-image bicubic view, nine overlapping 224px tiles; external native crop reported separately',
            'benign_controls':'Diagnostics only, not relabeled as authentic training images','human_verified':False}
        SIGNATURE=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        old=OUT/'protocol.json'
        if old.exists() and json.loads(old.read_text())['signature']!=SIGNATURE:raise ValueError('Different inputs or settings: use a new RUN_NAME and do not attach an incompatible resume ZIP.')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        if (OUT/'failure.json').exists():
            history=OUT/'history';history.mkdir(exist_ok=True);(OUT/'failure.json').replace(history/f'failure_{int(time.time())}.json')
        write_json(old,dict(config,signature=SIGNATURE))
        for name,frame in [('main',primary),('benign',benign),('external',external)]:frame.drop(columns=['local_path'],errors='ignore').to_csv(OUT/f'{name}_manifest.csv',index=False)
        write_json(OUT/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'GPU':torch.cuda.get_device_name(0),'token_saved':False})
        (OUT/'predictions').mkdir(exist_ok=True)
        pack('partial_inputs_verified')
        need_external=RUN_EXTERNAL and any(cached_features(name,'external',external) is None for name in MODELS)
        if need_external:external=external_pixels(external,metadata)
        repo=None;rows=previous_baselines(source,primary,external);device=torch.device('cuda');qualities=[95,75,50] if RUN_COMPRESSION else []
        for name,settings in MODELS.items():
            check();tasks=[('main',primary,None),('benign',benign,None)]+[('main',primary,q) for q in qualities]
            if RUN_EXTERNAL:tasks.append(('external',external,None))
            incomplete=[task for task in tasks if cached_features(name,task[0],task[1],task[2]) is None]
            model=None
            try:
                if incomplete:
                    if repo is None:repo=official_source()
                    attached=input_files(name+'.pth')+input_files('GenImage.pth' if settings['wsgm']==4 else 'training_setting_1.pth')
                    if len(attached)>1:raise ValueError('Multiple author checkpoints for '+name)
                    path=attached[0] if attached else download_file(settings['id'],CACHE/(name+'.pth'),settings['bytes'],settings['sha256'])
                    if sha(path)!=settings['sha256']:raise ValueError('Attached checkpoint differs from the audited official file')
                    say('Strict-loading verified author weights: '+name);model=load_detector(path,settings,repo,device)
                    probe=[Image.new('RGB',(224,224),(100,120,140))]*2
                    z=infer_views(model,probe,device)
                    if abs(z[0]-z[1])>1e-4:raise RuntimeError('Identical-input consistency preflight failed')
                    write_json(OUT/f'{name}_preflight.json',{'strict_checkpoint_load':True,'identical_input_consistency':True,'checkpoint_sha256':settings['sha256'],'not_a_study_result':True})
                    for kind,frame,quality in incomplete:extract_features(model,name,kind,frame,device,quality)
            finally:
                if model is not None:del model
                gc.collect();torch.cuda.empty_cache()
            features=cached_features(name,'main',primary);bf=cached_features(name,'benign',benign)
            ef=cached_features(name,'external',external) if RUN_EXTERNAL else None
            for evaluation,frame,scores in [('main',primary,features)]+([('external_repeated_challenge',external,ef)] if RUN_EXTERNAL else []):
                columns=[('OfficialCanonical','official_canonical_logit'),('WholeImageRaw','global_logit')]
                if evaluation!='main':columns.append(('OfficialNative','official_native_logit'))
                for label,column in columns:
                    p=expit(scores[column].to_numpy(dtype=np.float64));identifier=name+'_'+label
                    output=frame[['image_id','artifact_id','group','category','generator','label']].copy();output['probability']=p
                    output.to_csv(OUT/'predictions'/f'{identifier}_{evaluation}.csv',index=False,float_format='%.17g')
                    rows.append(dict(model=identifier,evaluation=evaluation,**metrics(frame.label,p,frame.group,frame.category)))
            jpeg={q:cached_features(name,'main',primary,q) for q in qualities}
            for kind in ['GlobalCal','GlobalPatchFusion']:
                say('Fitting small, group-disjoint development heads: '+name+' '+kind)
                rows+=adaptation(name,kind,primary,benign,external,features,bf,ef,jpeg)
            say('Computing regional measurements: '+name)
            patch_diagnostics(name,primary,features,source)
            pd.DataFrame(rows).to_csv(OUT/'summary_metrics.csv',index=False);pack('partial_model_comparison_complete')
        result=pd.DataFrame(rows);result.to_csv(OUT/'summary_metrics.csv',index=False);paired_comparisons(result)
        say('All declared comparisons completed.');pack('computational_complete_annotation_pending')
    except Exception as error:
        text=traceback.format_exc()
        if HF_TOKEN:text=text.replace(HF_TOKEN,'[REDACTED]')
        print(text,flush=True);write_json(OUT/'failure.json',{'traceback':text})
        pack('partial_time_limit' if isinstance(error,BudgetStop) else 'failed',str(error).replace(HF_TOKEN,'[REDACTED]') if HF_TOKEN else str(error))
    finally:
        stop.set();thread.join(timeout=2);HF_TOKEN=''
        print('OUTPUT:',ROOT/ZIP_NAME,flush=True)
        print('Download from the SAVED notebook version Output tab. Resume by attaching this output ZIP and rerunning the SAME cell and RUN_NAME.',flush=True)

if __name__=='__main__':run()
