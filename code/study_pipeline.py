"""RefundProof image construction and source-group evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import sys,time,json,hashlib,random,gc,traceback,zipfile,io,fnmatch,threading
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw,ImageFilter
import torch
import cv2
import joblib
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import evaluation_core as core
from download_transport import download,retry,HfApi
import download_transport
from region_audit import region_metrics,perturbation_audit

ROOT=Path(os.environ.get('RP_ROOT','/kaggle/working/RefundProof_FINAL'))
OUT=ROOT/'results'; CACHE=Path('/tmp/refundproof_model_cache')
SETTINGS=json.loads(os.environ.get('RP_SETTINGS','{}'))
DEFAULTS={'per_domain_cap':50,'canvas':512,'seeds':[42,123,777],
          'generation_steps':24,'max_generation_attempts':2,'benign_fraction':.25,
          'max_hours':10,'external_fraudbench':True,'patch_grid':4,
          'annotation_cases_per_domain':8,'generation_models':{
              'sd15':'stable-diffusion-v1-5/stable-diffusion-inpainting',
              'sdxl':'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'},
          'full_deep_transfer':True,'bootstrap':500,'version':'RefundProof-grounded-1.0'}
CFG={**DEFAULTS,**SETTINGS}
START=time.monotonic(); DEADLINE=START+float(CFG['max_hours'])*3600-300
API=None; REVISIONS={}; CURRENT='initializing'
SOURCE=None; ALL=None; PRIMARY=None; PATCH=None; EMB=None; FORENSIC=None; FAILURES=[]

def say(message):
    core.say(message)
def check():
    if time.monotonic()>DEADLINE:raise core.BudgetStop('Runtime allowance reached; save and reattach the output ZIP to resume.')
def save(path,data):core.write_json(path,data)
def hid(value):return core.stable(value)
def sha(path):return core.hash_file(path)
def rgb(path):
    with Image.open(path) as im:return ImageOps.exif_transpose(im).convert('RGB')
def square(image,size=512):
    return ImageOps.pad(image,(size,size),method=Image.Resampling.LANCZOS,color=(127,127,127),centering=(.5,.5))
def unload(*models):
    for model in models:
        if model is not None:
            try:model.to('cpu')
            except Exception:pass
    gc.collect();torch.cuda.empty_cache()
def pin(repo):
    if repo not in REVISIONS:
        REVISIONS[repo]=retry(lambda:API.model_info(repo).sha,repo+' revision')
        save(OUT/'MODEL_REVISIONS.json',REVISIONS)
    return REVISIONS[repo]
def model_path(repo,half=False):
    revision=pin(repo)
    if half:
        info=retry(lambda:API.model_info(repo,revision=revision),repo+' files')
        names=[f.rfilename for f in info.siblings]
        if not any('.fp16.safetensors' in n for n in names):
            raise RuntimeError(repo+' has no fp16 safetensors; refusing an unplanned full-weight fallback.')
        allowed=['*.json','*.txt','*.model','*.fp16.safetensors']
    else:allowed=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin']
    return download(repo,revision=revision,allow_patterns=allowed,local_dir=str(CACHE/hid(repo)))

OBJECTS={
 'Amazon_Fashion':['shirt','dress','jacket','shoe','bag','hat','scarf','pants','fabric'],
 'Cell_Phones_and_Accessories':['phone','phone case','charger','cable','headphones','screen','tripod'],
 'Electronics':['laptop','monitor','television','camera','speaker','headphones','keyboard','lamp','device'],
 'Grocery_and_Gourmet_Food':['food','bread','fruit','chips','bottle','can','jar','box','package'],
 'Home_and_Kitchen':['cup','plate','pan','bowl','bottle','chair','table','kettle','pillow','container','sticker'],
 'Tools_and_Home_Improvement':['tool','drill','handle','suitcase','box','pipe','metal object','container']}
RECIPES={
 'fabric':[('torn','a small realistic tear with frayed fabric fibres'),('stained','a visible localized irregular stain on the fabric')],
 'glass':[('cracked','a fine branching crack in the glass surface'),('scratched','several realistic fine surface scratches')],
 'metal':[('dented','a small physically plausible dent in the metal surface'),('scratched','a localized scuffed and scratched finish')],
 'food':[('spoiled','a small visible patch of mould or spoilage on the food')],
 'paper':[('torn','a localized tear in the packaging'),('crushed','a small crushed and creased section of packaging')],
 'plastic':[('cracked','a small localized crack in the plastic surface'),('scratched','localized scratches on the product surface')]}
def recipe_for(caption,object_name,domain,seed):

    text=object_name.lower()
    known=['shirt','dress','jacket','hat','scarf','pants','fabric','pillow','cloth','shoe','bag',
           'glass','cup','plate','bowl','screen','monitor','television','bread','fruit','chips','food','cake','vegetable','meat',
           'box','package','packaging','paper','carton','sticker','metal','pan','kettle','can','drill','tool','pipe']
    if not any(word in text for word in known):
        text+=' '+('glass' if 'glass' in caption.lower() else 'metal' if 'metal' in caption.lower() else '')
    if any(w in text for w in ['shirt','dress','jacket','hat','scarf','pants','fabric','pillow','cloth','shoe','bag']):material='fabric'
    elif any(w in text for w in ['glass','cup','plate','bowl','screen','monitor','television']):material='glass'
    elif any(w in text for w in ['bread','fruit','chips','food','cake','vegetable','meat']):material='food'
    elif any(w in text for w in ['box','package','packaging','paper','carton','sticker']):material='paper'
    elif any(w in text for w in ['metal','pan','kettle','can','drill','tool','pipe']):material='metal'
    else:material='plastic'
    damage,description=RECIPES[material][seed%len(RECIPES[material])]
    return material,damage,description

def discover_sources(input_root=Path('/kaggle/input')):
    global CURRENT
    CURRENT='source discovery and provenance'
    paths=sorted(Path(input_root).rglob('final_gold_with_text.csv'))
    if len(paths)!=1:raise ValueError('Expected exactly one final_gold_with_text.csv; found '+str(paths))
    csv_path=paths[0]; frame=pd.read_csv(csv_path,dtype=str).fillna('')
    frame.columns=[c.strip().lower().replace(' ','_') for c in frame.columns]
    for col in ['artifact_id','domain','image_path']:
        if col not in frame:raise ValueError('Missing source column '+col)
    if frame.artifact_id.duplicated().any():raise ValueError('Duplicate artifact IDs require upstream resolution.')
    if frame.artifact_id.eq('').any():raise ValueError('Empty artifact IDs.')
    if frame.artifact_id.str.contains(r'[/\\]|^\.{1,2}$',regex=True).any():raise ValueError('Unsafe artifact ID.')
    image_root=csv_path.parent.parent/'media dataset'
    if not image_root.exists():raise ValueError('Expected sibling media dataset folder next to final dataset.')
    files=sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'})
    bystem=defaultdict(list);byname=defaultdict(list)
    for path in files:bystem[path.stem].append(path);byname[path.name].append(path)
    records=[];exclusions=[]
    for row in frame.to_dict('records'):
        check();aid=row['artifact_id'];basename=Path(row['image_path'].replace('\\','/')).name
        candidates=bystem.get(aid,[]) or byname.get(basename,[])
        if len(candidates)>1:candidates=[p for p in candidates if row['domain'] in p.parts]
        if len(candidates)!=1:
            exclusions.append({'artifact_id':aid,'domain':row['domain'],'reason':'unresolved_or_ambiguous_path'});continue
        path=candidates[0]
        try:
            image=rgb(path)
            if min(image.size)<128:raise ValueError('short image edge below 128')
        except Exception as error:
            exclusions.append({'artifact_id':aid,'domain':row['domain'],'reason':'load_or_size_failure','detail':str(error)});continue
        product=row.get('parent_asin') or row.get('asin') or aid

        proxy='/'.join([row.get('user_id',''),product,row.get('timestamp','')])
        review=hid(proxy) if row.get('user_id') and row.get('timestamp') else hid(aid)
        records.append({'artifact_id':aid,'domain':row['domain'],'raw_path':str(path),
                        'raw_sha256':sha(path),'product_key':hid(product),'source_review_key':review,
                        'review_identity_scope':'user/product/timestamp proxy' if row.get('user_id') and row.get('timestamp') else 'artifact only',
                        'text':row.get('text',''),'title':row.get('title',''),
                        'historical_matched_label':row.get('matched_label','')})
    resolved=pd.DataFrame(records)
    if resolved.empty:raise ValueError('No source images resolved')

    resolved['selection_hash']=resolved.artifact_id.map(hid)
    resolved=resolved.sort_values('selection_hash').drop_duplicates('source_review_key')
    chosen=[]
    for domain,part in resolved.groupby('domain',sort=True):chosen.append(part.head(int(CFG['per_domain_cap'])))
    chosen=pd.concat(chosen).reset_index(drop=True)
    if len(chosen)<40:raise ValueError('Fewer than 40 source records; inspect provenance before training.')
    scientific_settings={k:v for k,v in CFG.items() if k not in {'max_hours','external_fraudbench','full_deep_transfer'}}
    signature=hid(json.dumps({'settings':scientific_settings,'csv_sha256':sha(csv_path),'pipeline_sha256':sha(__file__),
                             'core_sha256':sha(Path(core.__file__)),'sources':chosen[['artifact_id','raw_sha256']].to_dict('records')},sort_keys=True))
    previous=OUT/'study_configuration.json'
    if previous.exists() and json.loads(previous.read_text())['signature']!=signature:
        raise ValueError('Different code/configuration/input identity. Change RUN_NAME rather than mix caches.')
    CFG['signature']=signature;save(previous,CFG)
    chosen.to_csv(OUT/'source_candidates.csv',index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'source_exclusions.csv',index=False)
    stages=[('raw',frame),('resolved_one_per_review_proxy',resolved),('selected_before_visual_screen',chosen)]
    pd.concat([part.groupby('domain').size().rename('n').reset_index().assign(stage=name) for name,part in stages]).to_csv(OUT/'domain_distributions.csv',index=False)
    save(OUT/'stage_A_provenance.json',{'raw_csv_rows':len(frame),'candidate_image_files':len(files),'resolved_openable_rows':len(records),
         'unresolved_or_open_failure_rows':len(exclusions),'one_per_review_proxy_rows':len(resolved),'selected_before_visual_screen':len(chosen),
         'csv_sha256':sha(csv_path),'image_identity':'SHA256 of source bytes; decoded hashes added after normalization',
         'matched_label':'Preserved for audit only; never treated as verified damage',
         'review_grouping':'User/product/timestamp proxy when available; primary product groups strengthened by pixel and pHash duplicate components',
         'selection':'Smallest hashed source identity per review proxy; fixed per-domain cap independent of scores',
         'rights':'Source dataset provenance/rights supplied upstream, not newly certified by this pipeline'})
    return chosen

def make_edit_mask(object_mask,seed):
    rng=np.random.default_rng(seed);mask=np.asarray(object_mask,dtype=bool)
    if not mask.any():raise ValueError('Empty object mask')
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    candidates=np.argwhere(distance>=max(2,float(distance.max())*.25))
    if not len(candidates):raise ValueError('No stable object interior')
    cy,cx=candidates[rng.integers(len(candidates))]
    target=int(mask.sum()*rng.uniform(.15,.35));ratio=rng.uniform(.65,1.5)
    radius=max(4,np.sqrt(target/np.pi));yy,xx=np.indices(mask.shape)
    region=(((xx-cx)/(radius*ratio))**2+((yy-cy)/(radius/ratio))**2<=1)&mask
    if region.sum()<64:raise ValueError('Editing region too small')
    alpha=np.asarray(Image.fromarray(region.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(2)),dtype=np.float32)/255
    alpha*=mask
    return region,alpha

def prepare_objects(sources):
    global CURRENT
    CURRENT='captioning and product segmentation'
    from transformers import BlipProcessor,BlipForConditionalGeneration,AutoProcessor,AutoModelForZeroShotObjectDetection,SamProcessor,SamModel
    paths={'blip':model_path('Salesforce/blip-image-captioning-base'),
           'dino':model_path('IDEA-Research/grounding-dino-tiny'),'sam':model_path('facebook/sam-vit-base')}
    bp=BlipProcessor.from_pretrained(paths['blip']);bm=BlipForConditionalGeneration.from_pretrained(paths['blip']).cuda().eval()
    dp=AutoProcessor.from_pretrained(paths['dino']);dm=AutoModelForZeroShotObjectDetection.from_pretrained(paths['dino']).cuda().eval()
    sp=SamProcessor.from_pretrained(paths['sam']);sm=SamModel.from_pretrained(paths['sam']).cuda().eval()
    (OUT/'media'/'real').mkdir(parents=True,exist_ok=True);(OUT/'objects').mkdir(exist_ok=True)
    accepted=[];screen=[]
    for number,row in enumerate(sources.to_dict('records')):
        check();aid=row['artifact_id'];cache=OUT/'objects'/f'{aid}.json'
        if cache.exists():
            result=json.loads(cache.read_text());screen.append(result)
            if result.get('accepted'):accepted.append({**row,**result})
            continue
        try:
            image=square(rgb(row['raw_path']),CFG['canvas'])

            with torch.inference_mode():
                caption=bp.decode(bm.generate(**bp(images=image,return_tensors='pt').to('cuda'),max_new_tokens=40)[0],skip_special_tokens=True)
            if any(w in caption.lower() for w in ['screenshot','screen shot','text message','email','document','receipt','webpage']):
                raise ValueError('caption flagged screenshot/document')
            objects=OBJECTS.get(row['domain'],['product','object'])
            query='. '.join(objects)+'.'
            with torch.inference_mode():
                inputs=dp(images=image,text=query,return_tensors='pt').to('cuda');outputs=dm(**inputs)
            detection=dp.post_process_grounded_object_detection(outputs,inputs.input_ids,box_threshold=.30,text_threshold=.25,target_sizes=[image.size[::-1]])[0]
            valid=[]
            labels=detection.get('text_labels',detection.get('labels',[]))
            for box,confidence,name in zip(detection['boxes'],detection['scores'],labels):
                box=box.detach().cpu().numpy();area=max(0,box[2]-box[0])*max(0,box[3]-box[1])/image.width/image.height
                if .03<area<.95:valid.append((float(confidence),box,str(name)))
            if not valid:raise ValueError('No plausible target product box')
            confidence,box,name=max(valid,key=lambda v:v[0])
            sam_inputs=sp(image,input_boxes=[[box.tolist()]],return_tensors='pt').to('cuda')
            with torch.inference_mode():sam_outputs=sm(**sam_inputs)
            masks=sp.image_processor.post_process_masks(sam_outputs.pred_masks.cpu(),sam_inputs['original_sizes'].cpu(),sam_inputs['reshaped_input_sizes'].cpu())[0]
            best=int(sam_outputs.iou_scores[0,0].argmax());mask=masks[0,best].numpy().astype(bool)
            if mask.mean()<.02:raise ValueError('Target segmentation too small')
            seed=int(hid(aid)[:8],16)%2**31;region,alpha=make_edit_mask(mask,seed)
            real=OUT/'media/real'/f'{aid}.png';image.save(real)
            np.savez_compressed(OUT/'objects'/f'{aid}.npz',object_mask=mask,edit_region=region,alpha=alpha)
            material,damage,description=recipe_for(caption,name,row['domain'],seed)
            damage_cues=['broken','cracked','torn','stained','damaged','dented','mould','mold','rotten','shattered']
            source_condition='caption suggests visible damage; human verification pending' if any(cue in caption.lower() for cue in damage_cues) else 'no caption damage cue; human verification pending'
            result={'artifact_id':aid,'domain':row['domain'],'accepted':True,'caption':caption,'object_name':name,
                    'object_bbox':box.tolist(),'object_detector_score':confidence,'material_assumption':material,
                    'intended_damage':damage,'damage_description':description,'seed':seed,'real_path':str(real),
                    'edit_mask_fraction':float(region.mean()),'object_fraction':float(mask.mean()),
                    'eligibility_origin':'BLIP caption and Grounding DINO/SAM heuristics; human verification pending',
                    'source_damage_status':source_condition}
        except Exception as error:
            if isinstance(error,core.BudgetStop):raise
            if isinstance(error,torch.cuda.OutOfMemoryError):raise
            result={'artifact_id':aid,'domain':row['domain'],'accepted':False,'reason':str(error)}
        save(cache,result);screen.append(result)
        if result.get('accepted'):accepted.append({**row,**result})
        say(f'Product preparation {number+1}/{len(sources)}; retained {len(accepted)}; {aid}')
        if (number+1)%25==0:checkpoint('partial_product_preparation')
    unload(bm,dm,sm);del bm,dm,sm
    frame=pd.DataFrame(accepted);pd.DataFrame(screen).to_csv(OUT/'product_screening.csv',index=False)
    frame.to_csv(OUT/'prepared_sources.csv',index=False)
    if len(frame)<40:raise ValueError('Fewer than 40 usable product sources after automated screening. Inspect screening before final evaluation.')
    distribution=frame.groupby('domain').size().rename('n').reset_index().assign(stage='after_automated_product_screen')
    previous=pd.read_csv(OUT/'domain_distributions.csv');pd.concat([previous[previous.stage!='after_automated_product_screen'],distribution]).to_csv(OUT/'domain_distributions.csv',index=False)
    return frame

def generate(sources):
    global CURRENT
    CURRENT='localized damage generation and benign-edit controls'
    from diffusers import AutoPipelineForInpainting,DDIMScheduler
    (OUT/'generation').mkdir(exist_ok=True)
    plans=[]
    for row in sources.to_dict('records'):
        if str(row.get('source_damage_status','')).startswith('caption suggests visible damage'):
            continue
        for generator in CFG['generation_models']:
            plans.append((row,generator,'damage'))
            if int(hid(row['artifact_id']+'/benign')[:8],16)/2**32<CFG['benign_fraction']:
                plans.append((row,generator,'benign'))
    for generator,repo in CFG['generation_models'].items():
        timings=[]
        todo=[p for p in plans if p[1]==generator and not (OUT/'generation'/f'{p[0]["artifact_id"]}_{generator}_{p[2]}.json').exists()]
        if not todo:continue
        check();say('Loading '+repo)
        path=model_path(repo,half=True)
        pipe=AutoPipelineForInpainting.from_pretrained(path,torch_dtype=torch.float16,variant='fp16',use_safetensors=True)
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload();pipe.enable_attention_slicing();pipe.enable_vae_slicing()
        if hasattr(pipe,'enable_vae_tiling'):pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=False)
        size=1024 if generator=='sdxl' else 512
        for current_number,(row,_,kind) in enumerate(todo):
            check();started=time.monotonic();aid=row['artifact_id'];key=f'{aid}_{generator}_{kind}'
            source=rgb(row['real_path']);masks=np.load(OUT/'objects'/f'{aid}.npz');alpha=masks['alpha']
            instruction=row['damage_description'] if kind=='damage' else 'a small clean natural variation of the intact surface texture, with no damage or defect'
            prompt=f"Photorealistic product photograph: {row['caption']}. The {row['object_name']} has {instruction}. Keep the same object, material, shape, viewpoint, lighting and scene. Modify only the masked surface."
            negative='text, watermark, cartoon, drawing, new object, extra object, changed background, blurred photograph, unrealistic geometry'
            record={'artifact_id':aid,'generator':generator,'edit_kind':kind,'model_repo':repo,'model_revision':REVISIONS[repo],
                    'prompt':prompt,'negative_prompt':negative,'intended_damage':row['intended_damage'] if kind=='damage' else 'none',
                    'annotation_status':'automatic_pending_human_review','attempts':[],'ok':False}
            for attempt in range(CFG['max_generation_attempts']):
                check();seed=(row['seed']+attempt*104729+(0 if kind=='damage' else 7919))%2**31
                try:
                    core.reset(seed)
                    response=pipe(prompt=prompt,negative_prompt=negative,image=source.resize((size,size),Image.Resampling.LANCZOS),
                             mask_image=Image.fromarray((masks['edit_region']*255).astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST),
                             width=size,height=size,num_inference_steps=CFG['generation_steps'],strength=.95,guidance_scale=7.5,
                             generator=torch.Generator(device='cpu').manual_seed(seed))
                    if getattr(response,'nsfw_content_detected',None) and any(response.nsfw_content_detected):
                        raise ValueError('Generator flagged the output; rejected rather than count a blank image as damage')
                    raw=response.images[0]
                    if np.asarray(raw).std()<1:raise ValueError('Near-constant decoded generation; rejected')
                    generated=np.asarray(raw.resize(source.size,Image.Resampling.LANCZOS),dtype=np.float32)
                    original=np.asarray(source,dtype=np.float32)
                    composed=np.rint(original*(1-alpha[...,None])+generated*alpha[...,None]).clip(0,255).astype(np.uint8)
                    delta=np.abs(composed.astype(float)-original).max(axis=2)
                    support=alpha>0;changed=delta>8
                    meaningful=int((changed&support).sum());outside=float(delta[~support].max()) if (~support).any() else 0.
                    if meaningful<64:raise ValueError('Too few observable changed pixels')
                    if outside!=0:raise ValueError('Compositing changed unmasked pixels')
                    folder=OUT/'media'/f'{generator}_{kind}';folder.mkdir(exist_ok=True)
                    target=folder/f'{aid}.png';Image.fromarray(composed).save(target)
                    masks_folder=OUT/'changed_masks';masks_folder.mkdir(exist_ok=True)
                    mask_path=masks_folder/f'{key}.png';Image.fromarray(changed.astype(np.uint8)*255).save(mask_path)
                    record.update(ok=True,path=str(target),image_sha256=sha(target),changed_mask_path=str(mask_path),
                                  mask_provenance='Exact decoded pixel difference >8/255 after local compositing; computational change support, not human damage ground truth',
                                  changed_fraction=float(changed.mean()),outside_support_max_difference=outside,seed=seed,
                                  steps=CFG['generation_steps'],canvas=CFG['canvas'],model_canvas=size)
                    record['attempts'].append({'attempt':attempt+1,'seed':seed,'status':'passed_computational_checks'});break
                except Exception as error:
                    if isinstance(error,torch.cuda.OutOfMemoryError):raise
                    record['attempts'].append({'attempt':attempt+1,'seed':seed,'status':'failed','reason':str(error)})
            record['seconds']=time.monotonic()-started;save(OUT/'generation'/f'{key}.json',record)
            timings.append(record['seconds']);completed=len(list((OUT/'generation').glob('*.json')))
            say(f'Generation {completed}/{len(plans)}: {key}, ok={record["ok"]}, {record["seconds"]:.1f}s. Remaining for {generator}: approximately {(len(todo)-current_number-1)*np.mean(timings)/3600:.1f}h (rough).')
            if completed%20==0:checkpoint('partial_generation')
        unload(pipe);del pipe
    records=[json.loads(p.read_text()) for p in sorted((OUT/'generation').glob('*.json'))]
    pd.DataFrame([{k:v for k,v in r.items() if k!='attempts'} for r in records]).to_csv(OUT/'generation_manifest.csv',index=False)
    save(OUT/'stage_B_provenance.json',{'planned_variants':len(plans),'processed_variants':len(records),
       'successful_computational_edits':sum(r['ok'] for r in records),'failed_variants':sum(not r['ok'] for r in records),
       'models':CFG['generation_models'],'steps':CFG['generation_steps'],'benign_control_fraction':CFG['benign_fraction'],
       'object_mask':'Grounding DINO target box plus SAM segmentation','editing_mask':'Seeded region constrained to segmented object; exact alpha saved',
       'background_preservation':'All pixels outside alpha support retained exactly from normalized source',
       'authentic_processing':'Same 512x512 aspect-preserving letterbox and PNG encoding used for both classes',
       'human_verification':False,'semantic_damage_success':'Unverified; automated checks establish changed pixels and compositing only',
       'genuine_damage_candidate_controls':int(sources.source_damage_status.str.startswith('caption suggests visible damage').sum()),
       'selection_bias':'Model-assisted source screening and failed-generation exclusion; counts and reasons retained'})
    return records

def make_manifest(sources,generation):
    global CURRENT
    CURRENT='grouping, duplicate audit and evaluation manifest'
    good=[r for r in generation if r['ok'] and r['edit_kind']=='damage']

    bysource=defaultdict(set)
    for record in good:bysource[record['artifact_id']].add(record['generator'])
    complete={aid for aid,names in bysource.items() if names==set(CFG['generation_models'])}
    if len(complete)<40:raise ValueError('Fewer than 40 complete two-generator source sets; inspect failed generation before training.')
    controls=sources.get('source_damage_status',pd.Series('',index=sources.index)).str.startswith('caption suggests visible damage')
    selected=sources[sources.artifact_id.isin(complete)|controls].copy().reset_index(drop=True)
    parent=list(range(len(selected)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    hashes=[];phashes=[]
    for row in selected.to_dict('records'):
        im=rgb(row['real_path']);array=np.asarray(im)
        hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
        small=np.asarray(im.resize((32,32)).convert('L'),dtype=np.float32)
        dct=cv2.dct(small)[:8,:8].flatten();phashes.append(int.from_bytes(np.packbits(dct>np.median(dct[1:])).tobytes(),'big'))
    selected['pixel_sha256']=hashes;selected['phash']=phashes
    for column in ['product_key','source_review_key','pixel_sha256']:
        for indexes in selected.groupby(column).indices.values():
            for index in indexes[1:]:union(int(indexes[0]),int(index))
    for i,h in enumerate(phashes):
        for j,k in enumerate(phashes[:i]):
            if (h^k).bit_count()<=4:union(i,j)
    selected['group']=[hid('product_duplicate/'+selected.iloc[find(i)].artifact_id) for i in range(len(selected))]
    if selected.group.nunique()<20:raise ValueError('Fewer than 20 independent product/duplicate groups; final training is not justified.')
    if selected.groupby('group').size().max()>len(selected)*.25:raise ValueError('A duplicate/product component exceeds 25% of sources; inspect grouping.')
    records=[];edits={r['artifact_id']+'/'+r['generator']+'/'+r['edit_kind']:r for r in generation if r['ok']}
    for row in selected.to_dict('records'):
        aid=row['artifact_id']
        common={k:row[k] for k in ['artifact_id','product_key','source_review_key','group','text','title','phash']}
        common.update(category=row['domain'],source_path=row['real_path'],source_relative_path='media/real/'+aid+'.png',
                      source_damage_status=row.get('source_damage_status','unknown'),label_origin='unaltered source versus known controlled editing; source camera-authenticity and semantic damage unverified')
        records.append(dict(common,image_id=hid(aid+'/real'),generator='real',variant='real',path=row['real_path'],
                            relative_path='media/real/'+aid+'.png',label=0,eligible=True,damage='unknown',
                            changed_mask_path='',pixel_sha256=row['pixel_sha256'],sha256=sha(row['real_path'])))
        for generator in CFG['generation_models']:
            for kind in ['damage','benign']:
                record=edits.get(aid+'/'+generator+'/'+kind)
                if record is None:continue
                records.append(dict(common,image_id=hid(aid+'/'+generator+'/'+kind),generator=generator,variant=kind,path=record['path'],
                                    relative_path=str(Path(record['path']).relative_to(OUT)),label=1,eligible=kind=='damage',
                                    damage=row['intended_damage'] if kind=='damage' else 'none',changed_mask_path=record['changed_mask_path'],
                                    pixel_sha256=hashlib.sha256(np.asarray(rgb(record['path'])).tobytes()).hexdigest(),sha256=record['image_sha256']))
    frame=pd.DataFrame(records);primary=frame[frame.eligible].copy().reset_index(drop=True)
    frame.to_csv(OUT/'manifest_all.csv',index=False);primary.to_csv(OUT/'primary_manifest.csv',index=False)
    selected.to_csv(OUT/'complete_pair_sources.csv',index=False)
    save(OUT/'cohort_provenance.json',{'prepared_sources':len(sources),'complete_two_generator_sources':len(complete),
          'main_images':len(primary),'main_authentic':int((primary.label==0).sum()),'main_edited':int(primary.label.sum()),
          'benign_diagnostic_images':int((frame.variant=='benign').sum()),'independent_product_duplicate_groups':int(selected.group.nunique()),
       'unpaired_generation_exclusions':len(sources)-len(selected),'semantic_damage_labels':'intended generation recipes, unverified',
       'caption_flagged_genuine_damage_candidates':int(controls.sum()),
          'primary_group':'Product identity plus review proxy plus decoded-source equality and 64-bit pHash Hamming <=4',
          'text_ablation':'Same original text for authentic and all edited variants; no fabricated comments or class-coded text',
          'outcome':'controlled AI editing; not verified fraud intent or validated fabricated-damage claims'})
    return frame,primary

def extract_embeddings(frame):
    global CURRENT
    CURRENT='image and patch embeddings'
    from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
    from torchvision import transforms as T
    from transformers import CLIPModel,CLIPProcessor
    folder=OUT/'cache';folder.mkdir(exist_ok=True)
    fingerprint=hid(json.dumps(frame[['image_id','sha256']].to_dict('records'),sort_keys=True))
    guard=folder/'image_index.json'
    if guard.exists() and json.loads(guard.read_text())['fingerprint']!=fingerprint:raise ValueError('Feature cache image identity mismatch')
    save(guard,{'fingerprint':fingerprint,'image_ids':frame.image_id.tolist()})
    result={};patches=None
    norm=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    for name in ['efn','clip']:
        target=folder/f'{name}.npy';patch_path=folder/'patch_embeddings.npy'
        if target.exists() and (name!='efn' or patch_path.exists()):
            result[name]=np.load(target)
            if name=='efn':patches=np.load(patch_path)
            continue
        check();core.reset(42)
        if name=='efn':
            model=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1);model.classifier=torch.nn.Identity()
        else:
            path=model_path('openai/clip-vit-base-patch32');model=CLIPModel.from_pretrained(path);processor=CLIPProcessor.from_pretrained(path)
        model=model.cuda().eval();chunks=[];patch_chunks=[]
        with torch.inference_mode():
            for start in range(0,len(frame),16):
                check();images=[square(rgb(p),CFG['canvas']) for p in frame.path.iloc[start:start+16]]
                if name=='efn':
                    vectors=model(torch.stack([norm(im) for im in images]).cuda());chunks.append(vectors.cpu().numpy())
                    grid=CFG['patch_grid'];crops=[]
                    for im in images:
                        for y in range(grid):
                            for x in range(grid):crops.append(im.crop((x*im.width//grid,y*im.height//grid,(x+1)*im.width//grid,(y+1)*im.height//grid)))
                    patch_vectors=[]
                    for offset in range(0,len(crops),32):patch_vectors.append(model(torch.stack([norm(im) for im in crops[offset:offset+32]]).cuda()).cpu().numpy())
                    patch_chunks.append(np.concatenate(patch_vectors).reshape(len(images),grid*grid,-1))
                else:
                    inputs=processor(images=images,return_tensors='pt').to('cuda')

                    vectors=model.visual_projection(model.vision_model(pixel_values=inputs['pixel_values']).pooler_output)
                    vectors=vectors/vectors.norm(dim=1,keepdim=True).clamp_min(1e-8);chunks.append(vectors.cpu().numpy())
                if start%128==0:say(f'{name} embeddings: {min(start+16,len(frame))}/{len(frame)}')
        result[name]=np.concatenate(chunks);np.save(target,result[name])
        if name=='efn':patches=np.concatenate(patch_chunks);np.save(patch_path,patches)
        unload(model);del model
    return result,patches

def patch_targets(frame):
    grid=CFG['patch_grid'];labels=np.full((len(frame),grid*grid),-1,dtype=int)
    for i,row in frame.iterrows():
        if row.label==0:labels[i]=0;continue
        mask=np.asarray(rgb(row.changed_mask_path).convert('L'))>0
        h,w=mask.shape
        for y in range(grid):
            for x in range(grid):
                fraction=mask[y*h//grid:(y+1)*h//grid,x*w//grid:(x+1)*w//grid].mean()
                labels[i,y*grid+x]=1 if fraction>=.10 else 0 if fraction==0 else -1
    return labels

def patch_score(est,patches):
    values=est.decision_function(patches.reshape(-1,patches.shape[-1])).reshape(patches.shape[:2])

    return np.sort(values,axis=1)[:,-2:].mean(axis=1),values

def patch_fit(tr,te,seed,tag):
    check();key=hid(json.dumps(dict(tag,model='M6_patch'),sort_keys=True));base=OUT/'experiments'/key;base.mkdir(parents=True,exist_ok=True)
    if (base/'done.json').exists():return pd.read_csv(base/'predictions.csv')
    core.reset(seed);y=PRIMARY.label.to_numpy();g=PRIMARY.group.to_numpy()
    a,b,c=core.split_groups(y[tr],g[tr],seed);fit=tr[np.r_[a,b]];cal=tr[c]
    targets=patch_targets(PRIMARY.iloc[fit].reset_index(drop=True));features=PATCH[fit];keep=targets.reshape(-1)>=0
    patch_y=targets.reshape(-1)[keep]
    if len(np.unique(patch_y))<2:raise ValueError('No two-class patch supervision in training groups')
    est=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler()),
                  ('lr',LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=seed))])
    est.fit(features.reshape(-1,features.shape[-1])[keep],patch_y)
    cz,_=patch_score(est,PATCH[cal]);tz,heat=patch_score(est,PATCH[te]);sig=core.sigmoid_fit(cz,y[cal])
    p=core.calibrated(sig,tz);cp=core.calibrated(sig,cz)
    pred=core.export_predictions(te,expit(tz),p,dict(tag,model='M6_patch'));pred.to_csv(base/'predictions.csv',index=False)
    operational=core.operational_rows(y[cal],cp,y[te],p,dict(tag,model='M6_patch'))
    if tag['evaluation']=='main':operational=core.operating_uncertainty(operational,y[te],p,g[te])
    pd.DataFrame(operational).to_csv(base/'operational.csv',index=False)
    pd.DataFrame({'row_id':cal,'label':y[cal],'raw_logit':cz,'probability':cp}).to_csv(base/'calibration.csv',index=False)
    core.save_partition(base,fit,[],cal,te);joblib.dump((est,sig),base/'model.joblib',compress=3)
    if tag['evaluation']=='main' and seed==CFG['seeds'][0]:
        localization=[]
        for local,index in enumerate(te):
            row=PRIMARY.iloc[index]
            small=expit(heat[local]).reshape(CFG['patch_grid'],CFG['patch_grid'])
            target=base/'heatmaps';target.mkdir(exist_ok=True);np.save(target/f'{row.image_id}.npy',small)
            if row.label==1:
                mask=np.asarray(rgb(row.changed_mask_path).convert('L'))>0
                expanded=cv2.resize(small,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_LINEAR)
                localization.append(dict(image_id=row.image_id,artifact_id=row.artifact_id,group=row.group,generator=row.generator,
                                         annotation_target='computational_changed_pixel_support_not_verified_damage',**region_metrics(expanded,mask)))
        pd.DataFrame(localization).to_csv(base/'localization.csv',index=False)
    save(base/'done.json',{'model':'M6_patch','tag':tag,'signature':CFG['signature'],'training_supervision':'Exact computational changed-pixel support, no test masks used for prediction'})
    return pred

def setup_core(primary):
    core.ROOT=ROOT;core.OUT=OUT;core.PERSIST=None;core.API=API;core.REPO='';core.DATA=primary
    core.SEEDS=CFG['seeds'];core.DEADLINE=DEADLINE
    core.CONFIG.update(version=CFG['version'],dataset='attached RefundProof corpus',signature=CFG['signature'],
                       seeds=CFG['seeds'],bootstrap=CFG['bootstrap'],primary_group='product plus duplicate components',epochs=8,
                       caption_model='Salesforce/blip-image-captioning-base')
    core.budget=check

    def dataset(ids,quality=None):
        from torchvision import transforms as T
        transform=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        class Images(torch.utils.data.Dataset):
            def __len__(self):return len(ids)
            def __getitem__(self,i):
                row=core.DATA.iloc[ids[i]];im=square(core.load_image(row.path,quality),CFG['canvas'])
                return transform(im),torch.tensor(float(row.label),dtype=torch.float32)
        return Images()
    core.m5_dataset=dataset
    core.gradcam=gradcam

def gradcam(model,ids,probabilities,base):
    from torchvision import transforms as T
    norm=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    selected=[];metrics=[]

    frame=core.DATA.iloc[ids].copy();frame['index']=ids;frame['p']=probabilities
    for category,part in frame.groupby('category'):
        for label in [0,1]:
            candidates=part[part.label==label].sort_values('image_id')
            selected.extend(candidates.head(2).to_dict('records'))
    folder=base/'gradcam';folder.mkdir(exist_ok=True)
    for row in selected:
        check();im=square(rgb(row['path']),CFG['canvas']);xx=norm(im).unsqueeze(0).cuda();activation=[]
        def hook(module,args,out):out.retain_grad();activation.append(out)
        handle=model.features[-1].register_forward_hook(hook)
        try:
            model.zero_grad(set_to_none=True);model(xx)[0,0].backward();a=activation[0]
            heat=(a.grad.mean((2,3),keepdim=True)*a).sum(1).relu().detach().cpu().numpy()[0]
        finally:handle.remove()
        heat=cv2.resize(heat,(224,224),interpolation=cv2.INTER_LINEAR);heat=heat/(heat.max()+1e-8)
        np.save(folder/f'{row["image_id"]}.npy',heat)
        display=np.asarray(im.resize((224,224)))
        if row['label']==1:
            mask=np.asarray(rgb(row['changed_mask_path']).convert('L').resize((224,224),Image.Resampling.NEAREST))>0
            result=region_metrics(heat,mask)
            def predict(array):
                with torch.inference_mode():z=float(model(norm(Image.fromarray(array)).unsqueeze(0).cuda())[0,0])
                temperature=json.loads((base/'calibration.json').read_text())['temperature']
                return float(expit(z/temperature))
            if result['status']=='evaluated':
                perturb=perturbation_audit(display,heat,predict,repetitions=10,grid=8)
                for p in perturb:metrics.append(dict(image_id=row['image_id'],group=row['group'],target='computational_change_support',**result,**p))
    pd.DataFrame(metrics).to_csv(base/'gradcam_region_audit.csv',index=False)
    save(base/'gradcam_cases.json',[{k:r[k] for k in ['image_id','artifact_id','category','label','p']} for r in selected])

def experiments(all_frame,primary):
    global CURRENT,EMB,PATCH,FORENSIC
    CURRENT='feature extraction and source-disjoint detector experiments'
    setup_core(primary);all_features=core.get_forensic(all_frame)
    all_embeddings,all_patches=extract_embeddings(all_frame)
    lookup={v:i for i,v in enumerate(all_frame.image_id)};idx=[lookup[v] for v in primary.image_id]
    FORENSIC=all_features[idx];EMB={k:v[idx] for k,v in all_embeddings.items()};PATCH=all_patches[idx]
    paired_feature_analysis(all_frame,all_features)
    core.FQ={q:core.get_forensic(primary,q) for q in core.CONFIG['jpeg']}
    try:core.UFD=core.ufd_logits(primary)
    except Exception as error:
        if isinstance(error,core.BudgetStop):raise
        core.UFD=None;FAILURES.append({'stage':'pretrained_UFD','error':str(error)})
        save(OUT/'pretrained_baseline_failure.json',FAILURES[-1]);say('UFD comparison unavailable: '+str(error))
    tasks,names=core.plan_experiments()

    extra=[('M6_patch',tr,te,seed,tag) for name,tr,te,seed,tag in tasks if name=='M1' and tag['evaluation']!='cross_generator_source_disjoint']
    tasks+=extra
    if not CFG['full_deep_transfer']:
        tasks=[task for task in tasks if task[0]!='M5' or task[4]['evaluation']=='main']
        save(OUT/'deep_transfer_not_run.json',{'reason':'full_deep_transfer=False; explicitly not a complete deep-transfer study'})
    inputs={name:core.model_input(name,FORENSIC,EMB['efn'],EMB['clip'],primary.text.to_numpy()) for name in names if name!='M5'}
    save(OUT/'experiment_plan.json',{'experiments':len(tasks),'counts':dict(Counter(t[0]+' / '+t[4]['evaluation'] for t in tasks)),
                                   'semantic_damage_transfer':'Uses intended recipe labels only; human verification pending'})
    for number,(name,tr,te,seed,tag) in enumerate(tasks):
        check();say(f'Detector experiment {number+1}/{len(tasks)}: {name}, {tag}')
        if set(primary.group.iloc[tr])&set(primary.group.iloc[te]):raise ValueError('Group leakage')
        try:
            if name=='M5':core.m5(tr,te,seed,tag)
            elif name=='M6_patch':patch_fit(tr,te,seed,tag)
            else:core.shallow(name,inputs[name],tr,te,seed,tag)
        except Exception as error:
            if isinstance(error,core.BudgetStop):raise
            FAILURES.append({'stage':'detector','model':name,'tag':tag,'error':str(error),'traceback':traceback.format_exc()})
            save(OUT/'failed_experiments.json',FAILURES)
            if tag['evaluation']=='main':raise
            say('Transfer experiment failed, recorded: '+str(error))
        if (number+1)%10==0:checkpoint('partial_evaluation')
    benign_diagnostics(all_frame,all_features,all_embeddings,all_patches)

def paired_feature_analysis(frame,features):
    from scipy.stats import ttest_1samp
    real={r.artifact_id:i for i,r in frame[frame.label==0].iterrows()};rows=[];pairs=[]
    for (generator,variant),part in frame[frame.label==1].groupby(['generator','variant']):
        delta=[];groups=[]
        for index,row in part.iterrows():
            original=real.get(row.artifact_id)
            if original is None:raise ValueError('Edited image has no artifact-matched source')
            values=features[index]-features[original];delta.append(values);groups.append(row.group)
            pairs.append(dict(artifact_id=row.artifact_id,group=row.group,generator=generator,variant=variant,
                              **dict(zip(core.FEATURES,values))))
        paired=pd.DataFrame(delta,columns=core.FEATURES);paired['group']=groups
        group_means=paired.groupby('group')[core.FEATURES].mean()
        for feature in core.FEATURES:
            raw=paired[feature].dropna().to_numpy();values=group_means[feature].dropna().to_numpy();sd=float(raw.std(ddof=1)) if len(raw)>1 else 0.
            p=float(ttest_1samp(values,0).pvalue) if len(values)>1 and values.std()>0 else 1.
            rng=np.random.default_rng(42);boot=np.array([rng.choice(values,len(values)).mean() for _ in range(CFG['bootstrap'])])
            rows.append({'generator':generator,'variant':variant,'feature':feature,'artifact_pairs':len(raw),'independent_product_groups':len(values),
                         'mean_artifact_paired_delta':float(raw.mean()),'paired_artifact_dz':float(raw.mean()/sd) if sd else np.nan,
                         'mean_group_paired_delta':float(values.mean()),'group_mean_ci_lo':float(np.quantile(boot,.025)),
                         'group_mean_ci_hi':float(np.quantile(boot,.975)),'p':p,'inference_unit':'Product/duplicate group mean of artifact-matched differences'})
    result=pd.DataFrame(rows)
    if len(result):result['p_holm_all_features_generators_variants']=core.holm(result.p)
    result.to_csv(OUT/'paired_artifact_effects.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_feature_differences.csv',index=False)

def benign_diagnostics(frame,features,embeddings,patches):
    global CURRENT
    CURRENT='held-out benign-edit diagnostic'
    selected=frame[frame.variant=='benign'].copy();lookup={v:i for i,v in enumerate(frame.image_id)};rows=[]
    if selected.empty:
        save(OUT/'benign_controls_unavailable.json',{'reason':'No successful benign controls'});return
    for base in (OUT/'experiments').iterdir():
        done=base/'done.json'
        if not done.exists():continue
        meta=json.loads(done.read_text());tag=meta['tag'];name=meta['model']
        if tag['evaluation']!='main':continue
        partition=pd.read_csv(base/'split.csv');groups=set(partition.loc[partition.role=='test','group'])
        test=selected[selected.group.isin(groups)];idx=[lookup[v] for v in test.image_id]
        if not len(test):continue
        check()
        if name=='M5':
            model=core.m5_model();model.load_state_dict(torch.load(base/'best_weights.pt',map_location='cuda',weights_only=True))
            saved=core.DATA;core.DATA=test.reset_index(drop=True)
            try:z=core.m5_logits(model,np.arange(len(test)))
            finally:core.DATA=saved
            temp=json.loads((base/'calibration.json').read_text())['temperature'];p=expit(z/temp);unload(model);del model
        elif name=='M6_patch':
            est,sig=joblib.load(base/'model.joblib');z,_=patch_score(est,patches[idx]);p=core.calibrated(sig,z)
        elif name=='UFD':continue
        else:
            est,sig=joblib.load(base/'model.joblib');x=core.model_input(name,features[idx],embeddings['efn'][idx],embeddings['clip'][idx],test.text.to_numpy());p=core.calibrated(sig,core.score(est,x))
        d=test[['image_id','artifact_id','group','generator']].copy();d['model']=name;d['seed']=tag['seed'];d['probability_edit']=p
        d['interpretation']='Known benign AI edits; a positive authenticity flag is not evidence of a fabricated damage claim';rows.append(d)
    if rows:pd.concat(rows).to_csv(OUT/'benign_edit_predictions.csv',index=False)

def external_validation():
    global CURRENT,OUT
    CURRENT='external FraudBench validation'
    if not CFG['external_fraudbench']:
        save(OUT/'external_not_run.json',{'reason':'external_fraudbench=False'});return
    revision='27d94c1a4c4b087845f8a8e916473cdd6a5c6011'
    found=sorted(p for p in Path('/kaggle/input').rglob('DeepFake') if p.is_dir() and (p.parent/'Negative').exists())
    if found:dataset=found[0].parent.parent;say('Using attached FraudBench folder '+str(dataset))
    else:
        import shutil
        if shutil.disk_usage('/tmp').free<16*1024**3:raise RuntimeError('Less than 16 GiB temporary disk free for FraudBench. Attach the dataset separately and resume.')
        dataset=Path(download('TristanYan/FraudBench',repo_type='dataset',revision=revision,local_dir='/tmp/refundproof_fraudbench'))
    main_out=OUT;main_data=core.DATA;folder=main_out/'external';folder.mkdir(exist_ok=True)
    try:
        OUT=folder;core.OUT=folder
        all_external,test=core.audit_dataset(dataset)

        metadata_cache={};ecommerce=[]
        for row in test.to_dict('records'):
            source=Path(row['source_path']);mp=next(iter(sorted(source.parent.glob('MetaReview*.json'))),None)
            if mp not in metadata_cache:metadata_cache[mp]=json.loads(mp.read_text()) if mp else {}
            md=metadata_cache[mp];ecommerce.append(bool(md.get('asin') or md.get('parent_asin')))
        test=test[np.asarray(ecommerce)].copy().reset_index(drop=True)
        if test.empty:raise ValueError('No e-commerce sources identifiable by product metadata')
        main_sources=PRIMARY[PRIMARY.label==0]
        products=set(main_sources.product_key);pixels=set(main_sources.pixel_sha256);phashes=[int(v) for v in main_sources.phash]
        overlap=set();hash_cache={}
        for row in test.to_dict('records'):
            check();path=row['source_path']
            if path not in hash_cache:
                im=square(rgb(path),CFG['canvas']);array=np.asarray(im);pixel=hashlib.sha256(array.tobytes()).hexdigest()
                dct=cv2.dct(np.asarray(im.resize((32,32)).convert('L'),dtype=np.float32))[:8,:8].flatten()
                ph=int.from_bytes(np.packbits(dct>np.median(dct[1:])).tobytes(),'big');hash_cache[path]=(pixel,ph)
            pixel,ph=hash_cache[path]
            if row['product_key'] in products or pixel in pixels or any((ph^h).bit_count()<=4 for h in phashes):overlap.add(row['group'])
        before=len(test);test=test[~test.group.isin(overlap)].copy().reset_index(drop=True)
        if test.label.nunique()!=2:raise ValueError('External overlap removal left fewer than two classes')

        normalized=Path('/tmp/refundproof_external_normalized');normalized.mkdir(exist_ok=True)
        for index,row in test.iterrows():
            check();target=normalized/f'{row.image_id}.png'
            if not target.exists():square(rgb(row.path),CFG['canvas']).save(target)
            test.loc[index,'original_external_path']=row.path;test.loc[index,'path']=str(target);test.loc[index,'sha256']=sha(target)
        test.to_csv(folder/'external_manifest.csv',index=False)
        save(folder/'external_provenance.json',{'dataset':'TristanYan/FraudBench','revision':revision if not found else 'attached snapshot; verify upstream revision',
          'population':'E-commerce records identifiable by Amazon product metadata','before_overlap_removal':before,
          'after_overlap_removal':len(test),'excluded_overlap_groups':len(overlap),'genuine_damage_images':int((test.label==0).sum()),
          'AI_damage_images':int(test.label.sum()),'independent_groups':int(test.group.nunique()),
          'overlap_rules':'Original product IDs, normalized decoded-source equality, 64-bit pHash distance <=4; exclude entire source group',
          'label_origin':'FraudBench author annotations, not independently adjudicated cases',
          'model_selection':'Frozen development fold models; no external-label tuning or external calibration',
          'rights':'CC-BY-NC-SA-4.0; citation and license required','shared_source':'Both corpora include Amazon; overlap filtering does not establish a different acquisition platform'})
        features=core.get_forensic(test);embeddings,patches=extract_embeddings(test)
        ufd=None
        if core.UFD is not None:
            try:ufd=core.ufd_logits(test)
            except Exception as error:
                if isinstance(error,core.BudgetStop):raise
                save(folder/'ufd_failure.json',{'error':str(error)});FAILURES.append({'stage':'external_UFD','error':str(error)})
        core.DATA=test;predictions=[];operating=[]
        for base in sorted((main_out/'experiments').iterdir()):
            done=base/'done.json'
            if not done.exists():continue
            meta=json.loads(done.read_text());tag=meta['tag'];name=meta['model']
            if tag['evaluation']!='main':continue
            check();target=folder/'fold_predictions'/f'{base.name}.csv';target.parent.mkdir(exist_ok=True)
            op_target=folder/'fold_predictions'/f'{base.name}_operating.csv'
            if target.exists() and op_target.exists():
                predictions.append(pd.read_csv(target));operating.append(pd.read_csv(op_target));continue
            say(f'External frozen model: {name}, seed={tag["seed"]}, fold={tag["fold"]}')
            if name=='M5':
                model=core.m5_model();model.load_state_dict(torch.load(base/'best_weights.pt',map_location='cuda',weights_only=True))
                z=core.m5_logits(model,np.arange(len(test)));temp=json.loads((base/'calibration.json').read_text())['temperature'];p=expit(z/temp)
                unload(model);del model
            elif name=='M6_patch':
                est,sig=joblib.load(base/'model.joblib');z,_=patch_score(est,patches);p=core.calibrated(sig,z)
            elif name=='UFD':
                if ufd is None:continue
                _,sig=joblib.load(base/'model.joblib');z=ufd;p=core.calibrated(sig,z)
            else:
                est,sig=joblib.load(base/'model.joblib');x=core.model_input(name,features,embeddings['efn'],embeddings['clip'],test.text.to_numpy())
                z=core.score(est,x);p=core.calibrated(sig,z)
            frame=test[['image_id','artifact_id','group','category','generator','damage','label']].copy()
            frame['model']=name;frame['seed']=tag['seed'];frame['fold']=tag['fold'];frame['probability']=p;frame['raw_probability']=expit(z)
            frame.to_csv(target,index=False);predictions.append(frame)
            cal=pd.read_csv(base/'calibration.csv')
            op_rows=core.operational_rows(cal.label.to_numpy(),cal.probability.to_numpy(),test.label.to_numpy(),p,dict(tag,model=name,evaluation='external_frozen'))
            if name in ['M1','M5','M6_patch']:op_rows=core.operating_uncertainty(op_rows,test.label.to_numpy(),p,test.group.to_numpy())
            ops=pd.DataFrame(op_rows)
            ops.to_csv(op_target,index=False);operating.append(ops)
        if not predictions:raise ValueError('No external frozen models produced predictions')
        predictions=pd.concat(predictions);predictions.to_csv(folder/'predictions_all.csv',index=False)
        if operating:pd.concat(operating).to_csv(folder/'operating_points.csv',index=False)
        table=[]
        for name,part in predictions.groupby('model'):

            mean=part.groupby('image_id').probability.mean();joined=test.set_index('image_id').loc[mean.index]
            final=joined[['artifact_id','group','category','generator','label']].copy();final['probability']=mean;final.reset_index().to_csv(folder/f'final_probabilities_{name}.csv',index=False)
            table.append(dict(model=name,prediction_aggregation='Mean frozen development-fold calibrated probabilities',
                              **core.metrics(joined.label.to_numpy(),mean.to_numpy()),
                              **core.bootstrap_metrics(joined.label.to_numpy(),mean.to_numpy(),joined.group.to_numpy())))
        pd.DataFrame(table).to_csv(folder/'external_metrics.csv',index=False)
        if ufd is not None:
            pd.DataFrame([dict(model='UFD_pretrained_raw',**core.metrics(test.label.to_numpy(),expit(ufd)))]).to_csv(folder/'pretrained_raw_baseline_metrics.csv',index=False)
    finally:
        OUT=main_out;core.OUT=main_out;core.DATA=main_data

def export_results():
    global CURRENT
    CURRENT='aggregate metrics and regional measurements'
    if PRIMARY is not None:
        core.DATA=PRIMARY;core.OUT=OUT;core.collate('computational_results_annotation_pending')
    predictions=OUT/'predictions_all.csv'
    if PRIMARY is not None and predictions.exists():
        all_predictions=pd.read_csv(predictions);main=all_predictions[all_predictions.evaluation=='main'];table=[]
        for name,part in main.groupby('model'):
            completed=[group for _,group in part.groupby('seed') if len(group)==len(PRIMARY) and group.image_id.nunique()==len(PRIMARY)]
            if len(completed)!=len(CFG['seeds']):continue
            averaged=pd.concat(completed).groupby('image_id').probability.mean()
            truth=PRIMARY.set_index('image_id').loc[averaged.index]
            final=truth[['artifact_id','group','category','generator','damage','label']].copy();final['probability']=averaged
            final.reset_index().to_csv(OUT/f'final_probabilities_{name}.csv',index=False)
            table.append(dict(model=name,prediction_aggregation='Mean complete-seed cross-fitted calibrated probabilities',
                              **core.metrics(truth.label.to_numpy(),averaged.to_numpy()),
                              **core.bootstrap_metrics(truth.label.to_numpy(),averaged.to_numpy(),truth.group.to_numpy())))
        pd.DataFrame(table).to_csv(OUT/'main_final_ensemble_metrics.csv',index=False)
    localization=[];cam=[]
    for base in (OUT/'experiments').glob('*'):
        if (base/'done.json').exists():
            if (base/'localization.csv').exists():localization.append(pd.read_csv(base/'localization.csv'))
            if (base/'gradcam_region_audit.csv').exists():
                data=pd.read_csv(base/'gradcam_region_audit.csv')
                if not data.empty:cam.append(data)
    if localization:pd.concat(localization).to_csv(OUT/'patch_localization_change_support.csv',index=False)
    if cam:pd.concat(cam).to_csv(OUT/'gradcam_region_audit.csv',index=False)
    if (OUT/'patch_localization_change_support.csv').exists():
        data=pd.read_csv(OUT/'patch_localization_change_support.csv');valid=data[data.status=='evaluated']
        if len(valid):valid.groupby('generator')[['pointing_hit','attribution_mass_in_region','iou']].mean().reset_index().to_csv(OUT/'patch_region_summary.csv',index=False)
    if (OUT/'benign_edit_predictions.csv').exists():
        data=pd.read_csv(OUT/'benign_edit_predictions.csv')
        data.assign(flagged=data.probability_edit>=.5).groupby(['model','generator']).agg(
            mean_edit_probability=('probability_edit','mean'),fraction_flagged_as_edited=('flagged','mean'),n=('image_id','count')).reset_index().to_csv(OUT/'benign_edit_summary.csv',index=False)

def checkpoint(status,failure=None,final=False):
    global CURRENT
    if not OUT.exists():return
    save(OUT/'run_status.json',{'status':status,'stage':CURRENT,'failure':failure,'signature':CFG.get('signature'),
         'elapsed_hours':(time.monotonic()-START)/3600,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'human_annotation_verified':False,'next_step':'Inspect results and complete independent image/region validation before final damage-specific manuscript claims'})
    if final:
        hashes={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='checksums.json' and p.suffix!='.tmp'}
        save(OUT/'checksums.json',hashes)
    target=ROOT/'RefundProof_PAPER_OUTPUT.zip';temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix!='.tmp':archive.write(path,'RefundProof_PAPER/'+path.relative_to(OUT).as_posix())
    temporary.replace(target);say('Output ZIP checkpoint saved: '+str(target))

def restore():

    candidates=sorted(Path('/kaggle/input').rglob('RefundProof_PAPER_OUTPUT.zip'))
    if OUT.exists() and (OUT/'study_configuration.json').exists():return
    if len(candidates)>1:raise ValueError('Multiple resume output archives attached. Attach only the desired run.')
    if candidates:
        with zipfile.ZipFile(candidates[0]) as archive:
            for member in archive.infolist():
                parts=Path(member.filename).parts
                if not parts or parts[0]!='RefundProof_PAPER' or '..' in parts:raise ValueError('Unexpected resume archive structure')
                target=(OUT/Path(*parts[1:])).resolve()
                if not target.is_relative_to(OUT.resolve()):raise ValueError('Unsafe resume path')
                if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as source,target.open('wb') as dest:
                    import shutil;shutil.copyfileobj(source,dest)

def main():
    global API,REVISIONS,SOURCE,ALL,PRIMARY,CURRENT
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required for dataset construction and model training.')
    ROOT.mkdir(parents=True,exist_ok=True);restore();OUT.mkdir(parents=True,exist_ok=True);(OUT/'source').mkdir(exist_ok=True);(OUT/'experiments').mkdir(exist_ok=True)
    core.ROOT=ROOT;core.OUT=OUT;core.budget=check
    download_transport.BUDGET_HOOK=check
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.py','.md','.txt','.json']:
            import shutil;shutil.copy2(path,OUT/'source'/path.name)
    API=HfApi(token=os.environ['HF_TOKEN']);retry(lambda:API.whoami(),'Hugging Face read login')
    core.reset(42)
    revision_file=OUT/'MODEL_REVISIONS.json'
    if revision_file.exists():REVISIONS=json.loads(revision_file.read_text())
    save(OUT/'environment.json',{'python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,
         'gpu':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
         'packages':{d.metadata['Name']:d.version for d in __import__('importlib.metadata',fromlist=['distributions']).distributions()},
         'token_saved':False})
    import huggingface_hub;huggingface_hub.snapshot_download=download
    SOURCE=prepare_objects(discover_sources());checkpoint('partial_sources_complete')
    records=generate(SOURCE);ALL,PRIMARY=make_manifest(SOURCE,records);checkpoint('partial_generation_complete')
    experiments(ALL,PRIMARY)
    try:external_validation()
    except Exception as error:
        if isinstance(error,core.BudgetStop):raise
        FAILURES.append({'stage':'external_validation','error':str(error),'traceback':traceback.format_exc()})
        save(OUT/'external_failure.json',FAILURES[-1]);say('External validation incomplete: '+str(error))
    export_results();checkpoint('computational_complete_annotation_pending' if not FAILURES else 'partial_failed_analyses_annotation_pending',final=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(traceback.format_exc(),flush=True)
        OUT.mkdir(parents=True,exist_ok=True)
        try:
            if PRIMARY is not None:export_results()
        except Exception:save(OUT/'collation_failure.json',{'traceback':traceback.format_exc()})
        checkpoint('partial_time_limit' if isinstance(error,core.BudgetStop) else 'failed',str(error),final=True)
        sys.exit(2)
