
import os,sys,json,time,hashlib,traceback,shutil,gc,inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image,ImageFilter
from scipy.special import expit
from sklearn.metrics import log_loss
import study_pipeline as P
from region_audit import region_metrics

generator_source=inspect.getsource(P.generate)
old_prompt='prompt=f"Photorealistic product photograph: {row[\'caption\']}. The {row[\'object_name\']} has {instruction}. Keep the same object, material, shape, viewpoint, lighting and scene. Modify only the masked surface."'
new_prompt='prompt=f"Photorealistic {row[\'object_name\']} with {instruction}. Preserve overall product identity, material, viewpoint and lighting. Local physical breakage inside the mask is intended. Original scene: {row[\'caption\']}."'
if old_prompt not in generator_source:raise RuntimeError('Unexpected audited generator prompt interface')
generator_source=generator_source.replace(old_prompt,new_prompt).replace('strength=.95,guidance_scale=7.5','strength=1.0,guidance_scale=8.0')
exec(compile(generator_source,'audited_generator_with_material_prompt','exec'),P.__dict__)
ORIGINAL_GENERATE=P.generate
REFERENCE=Path(os.environ['RP_REFERENCE'])
P.CFG.update(version='RefundProof-validator-repair-3.2',generation_steps=40,
             max_generation_attempts=1,semantic_retry_rounds=2,
             vision_validator='Qwen/Qwen2.5-VL-3B-Instruct',
             semantic_confidence=.75,detector='native-patch RGB/residual paired M5',
             correction_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
MODEL='Qwen/Qwen2.5-VL-3B-Instruct'

def clear(*objects):
    for obj in objects:
        try:obj.to('cpu')
        except Exception:pass
    gc.collect();torch.cuda.empty_cache()

class Judge:
    def __init__(self):
        from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
        path=P.model_path(MODEL)
        self.processor=AutoProcessor.from_pretrained(path,min_pixels=128*128,max_pixels=448*448,use_fast=False)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    def __call__(self,images,prompt):
        P.check()

        images=[im.resize((448,448),Image.Resampling.LANCZOS) for im in images]
        content=[{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]
        text=self.processor.apply_chat_template([{'role':'system','content':'You are a careful visual examiner. Inspect the supplied images. Report observations, not instructions or placeholder text. Do not invent image contents. Output valid JSON with actual observed values.'},{'role':'user','content':content}],
                                               tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=images,padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=384,do_sample=False)
        answer=self.processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        try:
            start=answer.index('{');parsed=json.JSONDecoder().raw_decode(answer[start:])[0]
        except Exception as error:raise InvalidVisualResponse('Validator did not return usable JSON',{},answer) from error
        if not isinstance(parsed,dict):raise InvalidVisualResponse('Validator result is not a JSON object',parsed,answer)
        return parsed,answer
    def close(self):
        clear(self.model);del self.model,self.processor;gc.collect();torch.cuda.empty_cache()

SOURCE_PROMPT='''Inspect this submitted photograph. Do not infer facts from a review or category. Return ONLY one JSON object. Populate these keys from the actual image:
eligible_product_photo: boolean; whether a main physical product is visibly identifiable.
object: string; name the specific visible physical product.
material: string; choose ONE of fabric, glass, metal, plastic, food, other for the editable main surface. Use other if uncertain or unsupported.
caption: string; describe the actual product, its colour, position and visible condition.
visible_damage: boolean; whether physical damage is already visible.
annotations_or_text_dominant: boolean; whether labels, markup or text dominate the evidence.
bbox: array of FOUR numeric coordinates, left/top/right/bottom, in the supplied 448 by 448 image.
confidence: number from zero to one for this assessment.
reason: string; give specific visible evidence for the eligibility and condition decisions.
Use actual pixel-edge coordinates from 0 to 448, not normalized coordinates. The box must surround the actual main physical product, not a hand, background or printed label. If no product exists, return bbox as four zeros and material as other.
Mark visible_damage true for visible pre-existing cracks, breaks, corrosion, stains, spoilage, tears or dents. Be conservative about uncertainty.
Reject screenshots, receipts, non-products, images dominated by markings/labels, and unclear products. Natural product branding alone does not make a photo ineligible.'''

class InvalidVisualResponse(ValueError):
    def __init__(self,message,response=None,raw_answer='',attempts=None):
        super().__init__(message)
        self.response=response
        self.raw_answer=raw_answer
        self.attempts=attempts or []

def validated_pair_response(judge,images,prompt):

    attempts=[]
    for attempt in range(2):
        result={};answer=''
        try:
            query=prompt
            if attempt:
                query+='\nYour previous response did not satisfy the JSON schema: '+attempts[-1]['error']+'. Reinspect both images and report your own observations. Include every required key and a nonempty evidence sentence. Do not change a negative assessment to satisfy the requested edit.'
            result,answer=judge(images,query)
            validate_visual_response(result,'pair')
            attempts.append({'attempt':attempt+1,'response':result.copy(),'raw_answer':answer,'schema_valid':True})
            result=dict(result);result['response_attempts']=attempts
            return result,answer
        except (InvalidVisualResponse,ValueError) as error:
            attempts.append({'attempt':attempt+1,'response':getattr(error,'response',None) or result,
                'raw_answer':getattr(error,'raw_answer','') or answer,'schema_valid':False,'error':str(error)})
    last=attempts[-1]
    raise InvalidVisualResponse(last['error'],last['response'],last['raw_answer'],attempts)

def validator_control(judge,source=None):
    folder=P.OUT/'validator_controls';folder.mkdir(exist_ok=True)
    kind='blank_source' if source is None else 'identical_pair'
    if (folder/(kind+'.json')).exists():
        cached=json.loads((folder/(kind+'.json')).read_text())
        if cached.get('passed') is True:return
    result={};answer=''
    try:
        if source is None:
            result,answer=judge([Image.new('RGB',(512,512),(127,127,127))],SOURCE_PROMPT)
            validate_visual_response(result,'source')
            passed=result['eligible_product_photo'] is False and result['visible_damage'] is False and result['confidence']>=.75
        else:
            record={'path':source['real_path'],'edit_kind':'benign'}
            passed,result,answer=semantic_check(record,source,judge)
        P.save(folder/(kind+'.json'),{'passed':bool(passed),'response':result,'raw_answer':answer,
            'scope':'Known input consistency check; not human verification or benchmark validation'})
        if not passed:raise RuntimeError('Visual validator failed the '+kind+' control; no expensive generation will start.')
    except Exception as error:
        P.save(folder/(kind+'.json'),{'passed':False,'response':getattr(error,'response',None) or result,
            'raw_answer':getattr(error,'raw_answer','') or answer,'error':str(error),
            'response_attempts':getattr(error,'attempts',[])})
        raise

def validate_visual_response(result,kind):
    if not isinstance(result,dict):raise InvalidVisualResponse('Visual response is not an object')
    placeholders=('brief object name','factual short caption','brief evidence','describe exact visible change')
    text=json.dumps(result).lower()
    if any(value in text for value in placeholders):raise InvalidVisualResponse('Visual model copied a prompt placeholder')
    boolean_keys=(['eligible_product_photo','visible_damage','annotations_or_text_dominant'] if kind=='source'
                  else ['same_product','physical_damage_added','matches_requested_damage','realistic','unrelated_content_preserved'])
    if any(type(result.get(key)) is not bool for key in boolean_keys):raise InvalidVisualResponse('Missing or nonboolean visual decision')
    confidence=result.get('confidence')
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not 0<=confidence<=1:
        raise InvalidVisualResponse('Visual confidence is missing or outside zero to one')
    strings=['object','caption','reason'] if kind=='source' else ['evidence']
    if any(not isinstance(result.get(key),str) or len(result[key].strip())<3 for key in strings):
        missing=[key for key in strings if not isinstance(result.get(key),str) or len(result[key].strip())<3]
        raise InvalidVisualResponse('Missing actual visual description in key(s): '+', '.join(missing))
    if kind=='source':
        if result.get('material') not in ['fabric','glass','metal','plastic','food','other']:
            raise InvalidVisualResponse('Material must be one supported value, not a list or template')
        bbox=result.get('bbox')
        if not isinstance(bbox,list) or len(bbox)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in bbox):
            raise InvalidVisualResponse('Invalid numeric image bounding box')
        if any(x<0 or x>448 for x in bbox):raise InvalidVisualResponse('Image box is outside the 448-pixel coordinate system')
        if result['eligible_product_photo'] and (bbox[2]<=bbox[0] or bbox[3]<=bbox[1]):
            raise InvalidVisualResponse('Eligible product must have a box with positive area')
    return result

def abort_invalid_validator(records,kind):

    if len(records)>=10 and sum(bool(r.get('validator_response_invalid')) for r in records)/len(records)>.2:
        raise RuntimeError('Visual validator returned invalid/template responses for more than 20% of '+kind+' records. Saved diagnostics; stopped before wasting further generation time.')
RECIPES={
 'fabric':('torn','an unmistakable physical tear with an irregular open gap and clearly frayed fabric fibres; the actual fabric is ripped, not recoloured'),
 'glass':('cracked','a clearly visible branching fracture in the actual glass surface with realistic irregular crack edges and light reflections'),
 'metal':('dented','an unmistakable localized dent with visibly deformed metal and plausible highlights and shadows; not a printed pattern'),
 'plastic':('cracked','an unmistakable jagged physical crack and a small chipped edge in the actual plastic material'),
 'food':('spoiled','a clearly visible localized patch of realistic fuzzy mould or dark rotten tissue on the actual food, not on packaging or printed text')}

def edit_mask(object_mask,seed):
    rng=np.random.default_rng(seed);mask=np.asarray(object_mask,bool)
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)
    candidates=np.argwhere(distance>=max(2,float(distance.max())*.5))
    if not len(candidates):raise ValueError('No sufficiently wide product interior')
    cy,cx=candidates[rng.integers(len(candidates))]
    area=int(np.clip(round(mask.sum()*rng.uniform(.25,.45)),np.ceil(mask.sum()*.25),np.floor(mask.sum()*.45)))
    ratio=rng.uniform(.8,1.2);yy,xx=np.indices(mask.shape)
    object_positions=np.flatnonzero(mask)
    distance_rank=(((xx-cx)/ratio)**2+((yy-cy)*ratio)**2).ravel()[object_positions]
    selected=object_positions[np.argpartition(distance_rank,area-1)[:area]]
    region=np.zeros(mask.shape,bool);region.ravel()[selected]=True
    if region.sum()<256:raise ValueError('Insufficient editable product area')
    alpha=np.asarray(Image.fromarray(region.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(1)),np.float32)/255
    alpha*=mask
    return region,alpha

def prepare(sources):
    P.CURRENT='full-cohort product and pre-existing-damage screening'
    migration=P.OUT/'resume_source_candidates.csv'
    if migration.exists():
        prior=pd.read_csv(migration,dtype={'artifact_id':str,'raw_sha256':str})
        identity=lambda frame:sorted(zip(frame.artifact_id.astype(str),frame.raw_sha256.astype(str)))
        if identity(prior)!=identity(sources):raise ValueError('Preprocessing resume source IDs/bytes differ from the attached original dataset.')
        P.say('Verified all 300 cached source identities; continuing from saved product screening and masks.')
    runtime_preflight()
    folder=P.OUT/'source_quality';folder.mkdir(exist_ok=True)
    real=P.OUT/'media/real';real.mkdir(parents=True,exist_ok=True)
    pending=[r for r in sources.to_dict('records') if not (folder/(r['artifact_id']+'.json')).exists()]
    abort_invalid_validator([json.loads(p.read_text()) for p in folder.glob('*.json')],'source')
    if pending:
        judge=Judge()
        try:
            validator_control(judge)
            for row in pending:
                P.check();aid=row['artifact_id'];im=P.square(P.rgb(row['raw_path']),512);im.save(real/(aid+'.png'))
                answer='';result={}
                try:
                    result,answer=judge([im],SOURCE_PROMPT)
                    validate_visual_response(result,'source')
                    result.update(artifact_id=aid,raw_answer=answer,validator_revision=P.REVISIONS[MODEL])
                except Exception as error:
                    if isinstance(error,(P.core.BudgetStop,torch.cuda.OutOfMemoryError)):raise
                    result={'artifact_id':aid,'eligible_product_photo':False,'reason':str(error),'validator_response_invalid':True,
                            'raw_answer':answer,'unvalidated_response':json.dumps(result,default=str),'validator_revision':P.REVISIONS[MODEL]}
                P.save(folder/(aid+'.json'),result)
                done=len(list(folder.glob('*.json')));P.say(f'Full source screening {done}/{len(sources)}')
                if done%20==0:P.checkpoint('partial_full_source_screen')
                abort_invalid_validator([json.loads(p.read_text()) for p in folder.glob('*.json')],'source')
        finally:judge.close()
    from transformers import SamModel,SamProcessor
    processor=None;model=None;accepted=[];screen=[]
    (P.OUT/'objects').mkdir(exist_ok=True)
    try:
        for row in sources.to_dict('records'):
            P.check();aid=row['artifact_id'];target=P.OUT/'objects'/(aid+'.json')
            if target.exists():
                result=json.loads(target.read_text())
            else:
                result={'artifact_id':aid,'domain':row['domain'],'accepted':False}
                try:
                    decision=json.loads((folder/(aid+'.json')).read_text())
                    validate_visual_response(decision,'source')
                    if decision.get('eligible_product_photo') is not True or decision.get('annotations_or_text_dominant') is not False:
                        raise ValueError('Validator rejected source eligibility or annotation dominance')
                    if float(decision.get('confidence',0))<.75:raise ValueError('Source eligibility is uncertain')
                    if not isinstance(decision.get('visible_damage'),bool):raise ValueError('Source condition is uncertain')
                    material=decision.get('material','other').strip().lower()
                    if material not in RECIPES:raise ValueError('No supported physical-damage recipe for material')
                    bbox=np.asarray(decision['bbox'],float)*512/448
                    if bbox.shape!=(4,) or not np.isfinite(bbox).all():raise ValueError('Malformed product box')
                    bbox=np.clip(bbox,0,511)
                    if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:raise ValueError('Invalid product box')
                    if model is None:
                        path=P.model_path('facebook/sam-vit-base');processor=SamProcessor.from_pretrained(path)
                        model=SamModel.from_pretrained(path).cuda().eval()
                    image=P.rgb(real/(aid+'.png'));inputs=processor(image,input_boxes=[[bbox.tolist()]],return_tensors='pt').to('cuda')
                    with torch.inference_mode():output=model(**inputs)
                    masks=processor.image_processor.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu(),inputs['reshaped_input_sizes'].cpu())[0]
                    best=int(output.iou_scores[0,0].argmax());mask=masks[0,best].numpy().astype(bool)
                    if not .025<mask.mean()<.95:raise ValueError('Unusable product segmentation')
                    seed=int(P.hid(aid)[:8],16)%2**31;region,alpha=edit_mask(mask,seed)
                    np.savez_compressed(P.OUT/'objects'/(aid+'.npz'),object_mask=mask,edit_region=region,alpha=alpha)
                    damage,description=RECIPES[material]
                    result.update(accepted=True,caption=decision['caption'],object_name=decision['object'],
                                  object_bbox=bbox.tolist(),object_detector_score=float(decision['confidence']),
                                  object_bbox_provenance='Qwen sees an explicit 448x448 image; pixel coordinates are scaled by 512/448 for SAM',
                                  confidence_origin='MLLM self-reported confidence; not a calibrated detector probability',
                                  material_assumption=material,intended_damage=damage,damage_description=description,
                                  seed=seed,real_path=str(real/(aid+'.png')),edit_mask_fraction=float(region.mean()),
                                  edit_mask_object_fraction=float(region.sum()/mask.sum()),
                                  object_fraction=float(mask.mean()),eligibility_origin='Qwen2.5-VL structured screening and SAM; human verification pending',
                                  source_damage_status=('caption suggests visible damage; VLM-screened candidate; human verification pending'
                                      if decision['visible_damage'] else 'VLM suggests no visible damage; human verification pending'))
                except Exception as error:
                    if isinstance(error,(P.core.BudgetStop,torch.cuda.OutOfMemoryError)):raise
                    result['reason']=str(error)
                P.save(target,result)
            screen.append(result)
            if result.get('accepted'):accepted.append({**row,**result})
            P.say(f'Full product preparation {len(screen)}/{len(sources)}; retained {len(accepted)}')
            if len(screen)%25==0:P.checkpoint('partial_full_product_preparation')
    finally:
        if model is not None:clear(model);del model
    frame=pd.DataFrame(accepted);pd.DataFrame(screen).to_csv(P.OUT/'product_screening.csv',index=False)
    frame.to_csv(P.OUT/'prepared_sources.csv',index=False)
    if len(frame)<40:raise ValueError('Insufficient eligible sources. Quality report retained; no unsupported training result will be fabricated.')
    d=pd.read_csv(P.OUT/'domain_distributions.csv');d=d[d.stage.ne('after_automated_product_screen')]
    pd.concat([d,frame.groupby('domain').size().rename('n').reset_index().assign(stage='after_automated_product_screen')]).to_csv(P.OUT/'domain_distributions.csv',index=False)
    return frame

def semantic_check(record,row,judge):
    kind=record['edit_kind'];expected=row['intended_damage'] if kind=='damage' else 'no added physical damage'
    prompt='''Image 1 is the original and image 2 is a candidate. Compare BOTH images. They may be identical. Inspect the visible difference, not the requested intention. Return ONLY one JSON object conforming to this schema (the schema is NOT the output):
{"type":"object","required":["same_product","physical_damage_added","matches_requested_damage","realistic","unrelated_content_preserved","confidence","evidence"],"properties":{"same_product":{"type":"boolean"},"physical_damage_added":{"type":"boolean"},"matches_requested_damage":{"type":"boolean"},"realistic":{"type":"boolean"},"unrelated_content_preserved":{"type":"boolean"},"confidence":{"type":"number","minimum":0,"maximum":1},"evidence":{"type":"string","minLength":3}}}
same_product means both show the same physical product. physical_damage_added means a NEW physical defect is visible in image 2 that was absent in image 1. matches_requested_damage means the observed outcome matches the requested outcome. realistic means the product and any defect are physically plausible. unrelated_content_preserved means content outside the product edit remains intact.
The evidence value MUST be a full sentence describing the actual product in each image and the new difference and its location. If no new defect is visible, state that explicitly. Do not omit evidence, return null, or copy schema descriptions.
Requested outcome: '''+expected+'''. A texture change, colour shift, changed writing, blurred label, unchanged old defect or changed background is NOT added physical damage.
Be conservative: if the requested new physical damage cannot be clearly seen, set matches_requested_damage false. Assess realism of the actual defect.'''
    result,answer=validated_pair_response(judge,[P.rgb(row['real_path']),P.rgb(record['path'])],prompt)
    common=all(result.get(k) is True for k in ['same_product','realistic','unrelated_content_preserved']) and float(result.get('confidence',0))>=.75
    passed=common and (result.get('physical_damage_added') is True and result.get('matches_requested_damage') is True
                      if kind=='damage' else result.get('physical_damage_added') is False)
    return bool(passed),result,answer

def reuse_candidates(sources):
    marker=P.OUT/'old_candidate_reuse_complete.json'
    if marker.exists():return
    lookup={r['artifact_id']:r for r in sources.to_dict('records')}
    previous=REFERENCE/'generation'
    count=0
    for path in sorted(previous.glob('*.json')):
        record=json.loads(path.read_text());row=lookup.get(record['artifact_id'])
        if not row or not record.get('computational_ok',record.get('ok')) or str(row['source_damage_status']).startswith('caption suggests'):continue
        if record['edit_kind']=='damage' and record['intended_damage']!=row['intended_damage']:continue
        key=f"{record['artifact_id']}_{record['generator']}_{record['edit_kind']}"
        target=P.OUT/'generation'/(key+'.json')
        if target.exists():continue
        old_image=REFERENCE/'media'/f"{record['generator']}_{record['edit_kind']}"/(row['artifact_id']+'.png')
        old_mask=REFERENCE/'changed_masks'/(key+'.png')
        old_alpha=REFERENCE/'objects'/(row['artifact_id']+'.npz')
        if not old_image.exists() or not old_mask.exists() or not old_alpha.exists():continue
        original=REFERENCE/'media/real'/(row['artifact_id']+'.png')
        if not original.exists() or not np.array_equal(np.asarray(P.rgb(original)),np.asarray(P.rgb(row['real_path']))):continue
        if P.sha(old_image)!=record.get('image_sha256'):continue
        old_pixels=np.asarray(P.rgb(old_image));original_pixels=np.asarray(P.rgb(original))
        changed=np.abs(old_pixels.astype(float)-original_pixels).max(axis=2)>8
        recorded_mask=np.asarray(Image.open(old_mask).convert('L'))>0
        if not np.array_equal(changed,recorded_mask):continue
        current_masks=P.OUT/'objects'/(row['artifact_id']+'.npz')
        if not current_masks.exists():continue
        with np.load(current_masks) as masks:current_object=masks['object_mask'].astype(bool)
        overlap=float((changed&current_object).sum()/max(1,changed.sum()))
        if overlap<.95:continue
        image=P.OUT/'media'/f"{record['generator']}_{record['edit_kind']}"/(row['artifact_id']+'.png');image.parent.mkdir(parents=True,exist_ok=True)
        mask=P.OUT/'changed_masks'/(key+'.png');mask.parent.mkdir(exist_ok=True)
        alpha=P.OUT/'candidate_history'/(key+'_old_object_mask.npz');alpha.parent.mkdir(exist_ok=True)
        shutil.copy2(old_image,image);shutil.copy2(old_mask,mask);shutil.copy2(old_alpha,alpha)
        record.update(path=str(image),changed_mask_path=str(mask),prior_run_candidate=True,ok=True,
                      editing_alpha_provenance=str(alpha.relative_to(P.OUT)),semantic_checked=False,semantic_pass=False,
                      reused_change_overlap_with_current_product=overlap,
                      validator_response_invalid=False)
        P.save(target,record);count+=1
    P.save(marker,{'copied_candidates':count,'selection':'Matching recipe and computational success only; every retained old candidate receives new semantic screening'})

def check_candidates(sources,round_number):
    lookup={r['artifact_id']:r for r in sources.to_dict('records')}
    pending=[p for p in sorted((P.OUT/'generation').glob('*.json')) if not json.loads(p.read_text()).get('semantic_checked')]
    abort_invalid_validator([json.loads(p.read_text()) for p in (P.OUT/'generation').glob('*.json') if json.loads(p.read_text()).get('computational_ok') and json.loads(p.read_text()).get('semantic_checked')],'pair')
    if not pending:return
    P.CURRENT='full-cohort pairwise semantic quality screening';judge=Judge()
    try:
        for number,path in enumerate(pending):
            P.check();record=json.loads(path.read_text());passed=False;decision={};answer=''
            if record.get('ok'):
                try:passed,decision,answer=semantic_check(record,lookup[record['artifact_id']],judge)
                except Exception as error:
                    if isinstance(error,(P.core.BudgetStop,torch.cuda.OutOfMemoryError)):raise
                    answer=getattr(error,'raw_answer','')
                    decision={'error':str(error),'validator_response_invalid':True,
                        'unvalidated_response':getattr(error,'response',None),'response_attempts':getattr(error,'attempts',[])}
            record.update(computational_ok=bool(record.get('ok')),ok=passed,semantic_checked=True,
                          semantic_round=round_number,semantic_pass=passed,semantic_decision=decision,
                          semantic_raw_answer=answer,semantic_validator=MODEL,
                          validator_response_invalid=bool(decision.get('validator_response_invalid')),
                          semantic_validator_revision=P.REVISIONS[MODEL],
                          annotation_status='MLLM screened; independent human verification pending')
            P.save(path,record);P.say(f'Full semantic screening {number+1}/{len(pending)}; {path.stem}: accepted={passed}')
            if (number+1)%20==0:P.checkpoint('partial_full_semantic_screen')
            abort_invalid_validator([json.loads(p.read_text()) for p in (P.OUT/'generation').glob('*.json') if json.loads(p.read_text()).get('computational_ok') and json.loads(p.read_text()).get('semantic_checked')],'pair')
    finally:judge.close()

def generation(sources):
    (P.OUT/'generation').mkdir(exist_ok=True)
    controls=P.OUT/'validator_controls/identical_pair.json'
    if not controls.exists() or not json.loads(controls.read_text()).get('passed'):
        judge=Judge()
        try:validator_control(judge,sources.iloc[0].to_dict())
        finally:judge.close()
    reuse_candidates(sources);check_candidates(sources,-1)
    for round_number in range(P.CFG['semantic_retry_rounds']):
        marker=P.OUT/f'generation_round_{round_number}_complete.json'
        if marker.exists():continue

        for path in sorted((P.OUT/'generation').glob('*.json')):
            record=json.loads(path.read_text())
            if record.get('semantic_checked') and not record.get('semantic_pass') and record.get('semantic_round',-1)<round_number:
                history=P.OUT/'candidate_history'/f'{path.stem}_round_{record.get("semantic_round",-1)}';history.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,history/'record.json')
                for field in ['path','changed_mask_path']:
                    source=Path(record.get(field,''))
                    if source.is_file():shutil.copy2(source,history/(field+'.png'))
                path.unlink()
        attempt_sources=sources.copy();attempt_sources['seed']=(attempt_sources.seed+104729*(round_number+1))%(2**31)
        ORIGINAL_GENERATE(attempt_sources);check_candidates(sources,round_number)
        P.save(marker,{'round':round_number,'completed':True})
        P.checkpoint('partial_semantic_generation_round_complete')
    records=[json.loads(p.read_text()) for p in sorted((P.OUT/'generation').glob('*.json'))]
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['attempts','semantic_decision','semantic_raw_answer']} for r in records]).to_csv(P.OUT/'generation_manifest.csv',index=False)
    P.save(P.OUT/'semantic_quality_provenance.json',{'validator':MODEL,'revision':P.REVISIONS[MODEL],
        'full_candidate_records':len(records),'passed_computational_and_MLLM_checks':sum(r.get('ok',False) for r in records),
        'human_verified':False,'retry_rounds':P.CFG['semantic_retry_rounds'],
        'selection':'Source quality and visible-damage checks occur before detector outcomes; no selection by detector score'})
    provenance=json.loads((P.OUT/'stage_B_provenance.json').read_text())
    provenance.update(successful_computational_edits=sum(r.get('computational_ok',False) for r in records),
        MLLM_screened_passes=sum(r.get('ok',False) for r in records),
        object_mask='Qwen2.5-VL product box with SAM refinement; reused candidates retain their recorded original alpha',
        editing_mask='Seeded object-interior region and saved alpha; actual fractions recorded per source',
        semantic_damage_success='MLLM-screened only; independent human verification pending')
    P.save(P.OUT/'stage_B_provenance.json',provenance)
    return records

class NativePatchDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
        backbone=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features=backbone.features
        for block in list(self.features.children())[:-2]:
            for parameter in block.parameters():parameter.requires_grad=False
        self.residual=torch.nn.Sequential(torch.nn.Conv2d(3,16,3,padding=1),torch.nn.ReLU(),
            torch.nn.Conv2d(16,32,3,stride=2,padding=1),torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1))
        self.head=torch.nn.Linear(1280+64,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def patch_logits(self,images):
        b=images.shape[0]
        patches=images.unfold(2,128,128).unfold(3,128,128).permute(0,2,3,1,4,5).reshape(b*16,3,128,128)
        rgb=self.features((patches-self.mean)/self.std).mean((2,3))
        residual=patches-torch.nn.functional.avg_pool2d(patches,3,stride=1,padding=1)
        noise=self.residual(residual).flatten(1)
        return self.head(torch.cat([rgb,noise],1)).reshape(b,16)
    def forward(self,images):return self.patch_logits(images).topk(2,dim=1).values.mean(1,keepdim=True)

def image_tensor(path,quality=None):
    from torchvision.transforms.functional import to_tensor
    return to_tensor(P.square(P.core.load_image(path,quality),512))

def image_dataset(ids,quality=None):
    class Images(torch.utils.data.Dataset):
        def __len__(self):return len(ids)
        def __getitem__(self,i):
            row=P.core.DATA.iloc[ids[i]]
            return image_tensor(row.path,quality),torch.tensor(float(row.label))
    return Images()

def runtime_preflight():
    P.CURRENT='dependency and CUDA training preflight before image generation'
    P.OUT.mkdir(exist_ok=True)
    P.say('Checking model classes, grouped statistics and a full-size native-patch training batch before generation.')
    from transformers import Qwen2_5_VLForConditionalGeneration,SamModel,SamProcessor
    from diffusers import AutoPipelineForInpainting,DDIMScheduler
    C=P.core;C.reset(42)
    labels=np.array([0,0,1,1]);probabilities=np.array([.5,.5,.5,.9])
    for target in [0,.01,.05,.1]:
        threshold=C.fpr_threshold(labels,probabilities,target)
        if C.rates(labels,probabilities,threshold)['fpr']>target:raise RuntimeError('Exact FPR threshold preflight failed')
    model=None;optimizer=None
    try:
        model=NativePatchDetector().cuda();model.train()
        for block in list(model.features.children())[:-2]:block.eval()

        images=torch.rand(6,3,512,512,device='cuda')
        targets=torch.zeros(6,16,device='cuda');targets[[1,2,4,5],5]=1
        y=torch.tensor([0.,1.,1.,0.,1.,1.],device='cuda')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        patch=model.patch_logits(images);image_score=patch.topk(2,dim=1).values.mean(1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(image_score,y)+.5*torch.nn.functional.binary_cross_entropy_with_logits(patch,targets)
        loss=loss+.25*torch.nn.functional.softplus(.5-(patch[1,5]-patch[0,5]))
        loss.backward();optimizer.step()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite native-patch CUDA training preflight')
        model.eval()
        with torch.inference_mode():prediction=model(images[:1])
        if prediction.shape!=(1,1) or not torch.isfinite(prediction).all():raise RuntimeError('Native-patch inference preflight failed')
        P.save(P.OUT/'runtime_preflight.json',{'passed':True,'batch_images':6,'patches_per_image':16,
            'synthetic_fixture_only':True,'pretrained_vision_validator_grounding_verified':False,
            'scope':'Actual CUDA forward/backward/optimizer and imported dependency interfaces; not research metrics'})
        del images,targets,y,patch,image_score,loss,prediction
    finally:
        if model is not None:clear(model)
        del model,optimizer;gc.collect();torch.cuda.empty_cache();C.reset(42)
    P.CURRENT='full-cohort product and pre-existing-damage screening'
    P.say('CUDA training preflight passed; continuing with all selected source candidates.')

def train_detector(tr,te,seed,tag):
    P.check();C=P.core;key=C.stable(json.dumps(dict(tag,model='M5'),sort_keys=True));base=P.OUT/'experiments'/key;base.mkdir(parents=True,exist_ok=True)
    if (base/'done.json').exists():return pd.read_csv(base/'predictions.csv')

    if tag['evaluation']=='cross_generator_source_disjoint' and tag['source']!=tag['target']:
        source_tag={'evaluation':'LOGO_source_disjoint','seed':seed,'held_out':tag['target'],'fold':tag['fold']}
        source_base=P.OUT/'experiments'/C.stable(json.dumps(dict(source_tag,model='M5'),sort_keys=True))
        if (source_base/'done.json').exists():
            meta=json.loads((source_base/'done.json').read_text());split=pd.read_csv(source_base/'split.csv')
            saved_train=set(split.loc[split.role!='test','image_id']);saved_test=set(split.loc[split.role=='test','image_id'])
            if meta.get('signature')==P.CFG['signature'] and saved_train==set(C.DATA.iloc[tr].image_id) and saved_test==set(C.DATA.iloc[te].image_id):
                shutil.copytree(source_base,base,dirs_exist_ok=True,ignore=shutil.ignore_patterns('done.json'))
                for filename in ['predictions.csv','operational.csv']:
                    table=pd.read_csv(base/filename).drop(columns=['held_out'],errors='ignore')
                    for field,value in tag.items():table[field]=value
                    table.to_csv(base/filename,index=False)
                P.save(base/'done.json',{'model':'M5','tag':tag,'signature':P.CFG['signature'],
                    'reused_identical_partition_from':source_base.name,'partition_identity_checked':True})
                return pd.read_csv(base/'predictions.csv')
    C.reset(seed);data=C.DATA;y=data.label.to_numpy();g=data.group.to_numpy()
    a,b,c=C.split_groups(y[tr],g[tr],seed);fit,val,cal=tr[a],tr[b],tr[c];C.save_partition(base,fit,val,cal,te)
    class Bundles(torch.utils.data.Dataset):
        def __init__(self):self.groups=list(data.iloc[fit].groupby('artifact_id').indices.values())
        def __len__(self):return len(self.groups)
        def __getitem__(self,i):
            indexes=fit[self.groups[i]];xs=[];ys=[];targets=[]
            for index in indexes:
                row=data.iloc[index];xs.append(image_tensor(row.path));ys.append(float(row.label))
                if row.label:
                    mask=np.asarray(Image.open(row.changed_mask_path).convert('L'))>0
                    fraction=np.array([mask[j*128:(j+1)*128,k*128:(k+1)*128].mean() for j in range(4) for k in range(4)])
                    targets.append(np.where(fraction>=.015,1,np.where(fraction==0,0,-1)))
                else:targets.append(np.zeros(16))
            return torch.stack(xs),torch.tensor(ys),torch.tensor(np.asarray(targets),dtype=torch.float32)
    def collate(items):
        sizes=[len(item[1]) for item in items]
        return torch.cat([v[0] for v in items]),torch.cat([v[1] for v in items]),torch.cat([v[2] for v in items]),sizes
    model=NativePatchDetector().cuda();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    best=float('inf');stale=0;history=[]
    for epoch in range(12):
        P.check();model.train()
        for block in list(model.features.children())[:-2]:block.eval()
        dl=torch.utils.data.DataLoader(Bundles(),batch_size=2,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(seed+epoch))
        losses=[]
        for images,labels,targets,sizes in dl:
            P.check();images=images.cuda();labels=labels.cuda();targets=targets.cuda();opt.zero_grad(set_to_none=True)
            patch=model.patch_logits(images);logits=patch.topk(2,dim=1).values.mean(1)
            image_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
            known=targets>=0
            patch_loss=torch.nn.functional.binary_cross_entropy_with_logits(patch[known],targets[known])
            ranking=[];offset=0
            for size in sizes:
                zero=torch.where(labels[offset:offset+size]==0)[0]
                if len(zero):
                    reference=patch[offset+int(zero[0])]
                    for index in range(offset,offset+size):
                        changed=targets[index]==1
                        if labels[index]==1 and changed.any():ranking.append(torch.nn.functional.softplus(.5-(patch[index,changed]-reference[changed])).mean())
                offset+=size
            rank=torch.stack(ranking).mean() if ranking else patch.sum()*0
            loss=image_loss+.5*patch_loss+.25*rank;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        z=C.m5_logits(model,val);validation=float(log_loss(y[val],expit(z),labels=[0,1]))
        history.append(dict(epoch=epoch+1,training_loss=np.mean(losses),validation_loss=validation))
        pd.DataFrame(history).to_csv(base/'training_history.csv',index=False)
        P.say(f'Native patch detector {tag} epoch {epoch+1}/12, val loss={validation:.4f}')
        if validation<best-1e-5:
            best=validation;stale=0;torch.save(model.state_dict(),base/'best_weights.pt')
        else:stale+=1
        if stale>=3:break
    model.load_state_dict(torch.load(base/'best_weights.pt',map_location='cuda',weights_only=True))
    cz=C.m5_logits(model,cal);temp=C.temperature(cz,y[cal]);tz=C.m5_logits(model,te);cp=expit(cz/temp);p=expit(tz/temp)
    prediction=C.export_predictions(te,expit(tz),p,dict(tag,model='M5'));prediction.to_csv(base/'predictions.csv',index=False)
    pd.DataFrame(dict(row_id=cal,label=y[cal],raw_logit=cz,probability=cp)).to_csv(base/'calibration.csv',index=False)
    rows=C.operational_rows(y[cal],cp,y[te],p,dict(tag,model='M5'))
    if tag['evaluation']=='main':rows=C.operating_uncertainty(rows,y[te],p,g[te])
    pd.DataFrame(rows).to_csv(base/'operational.csv',index=False)
    P.save(base/'calibration.json',dict(temperature=temp,early_stop_best_loss=best,architecture='native 4x4 RGB/residual patch model with training-only mask and paired ranking losses'))
    if tag['evaluation']=='main':
        for quality in C.CONFIG['jpeg']:
            z=C.m5_logits(model,te,quality);C.export_predictions(te,expit(z),expit(z/temp),dict(tag,model='M5',jpeg_quality=quality)).to_csv(base/f'jpeg_{quality}.csv',index=False)
        if seed==C.SEEDS[0]:
            heatdir=base/'native_patch_maps';heatdir.mkdir(exist_ok=True);metrics=[]
            for index in te:
                P.check();row=data.iloc[index]
                with torch.inference_mode():heat=model.patch_logits(image_tensor(row.path)[None].cuda()).cpu().numpy()[0]
                heat=expit(heat).reshape(4,4);np.save(heatdir/(row.image_id+'.npy'),heat)
                expanded=cv2.resize(heat,(512,512),interpolation=cv2.INTER_LINEAR)
                if row.label:
                    mask=np.asarray(Image.open(row.changed_mask_path).convert('L'))>0
                    metrics.append(dict(image_id=row.image_id,artifact_id=row.artifact_id,generator=row.generator,group=row.group,annotation_target='computational change support; not human damage ground truth',**region_metrics(expanded,mask)))
            pd.DataFrame(metrics).to_csv(base/'native_patch_region_metrics.csv',index=False)
    P.save(base/'done.json',dict(model='M5',tag=tag,signature=P.CFG['signature'],model_version='paired RGB/residual native patch v2'))
    clear(model);del model,opt
    return prediction

ORIGINAL_SETUP=P.setup_core
ORIGINAL_PLAN=P.core.plan_experiments
def plan_experiments():
    tasks,names=ORIGINAL_PLAN()
    tasks += [('M5',tr,te,seed,dict(tag)) for name,tr,te,seed,tag in list(tasks)
              if name=='M1' and tag['evaluation']=='cross_generator_source_disjoint']
    return tasks,names

def setup(data):
    ORIGINAL_SETUP(data);C=P.core;C.CONFIG.update(batch_size=4,epochs=12,patience=3,
        m5='Native 4x4 RGB/residual patches, ImageNet backbone last two blocks tuned, training-only mask and paired ranking supervision')
    C.m5_model=lambda:NativePatchDetector().cuda();C.m5_dataset=image_dataset;C.m5=train_detector
    C.plan_experiments=plan_experiments

    def loader(ids,shuffle=False,seed=42,quality=None):
        return torch.utils.data.DataLoader(image_dataset(ids,quality),batch_size=4,shuffle=shuffle,
            num_workers=0,generator=torch.Generator().manual_seed(seed),pin_memory=True)
    C.loader=loader

def migrate_extracted_preprocessing(folder):
    configuration=json.loads((folder/'study_configuration.json').read_text())
    if hashlib.sha256((folder/'source/correction.py').read_bytes()).hexdigest()!='b192ed834a2c9df4a929a07e5e35c7b085215f0e34ab340cfc002a51d65f6da7':
        raise ValueError('Preprocessing migration requires the exact audited v3.1 worker.')
    if any(p.is_file() for name in ['generation','experiments'] for p in (folder/name).rglob('*')):
        raise ValueError('Only pre-generation v3.1 work can be migrated.')
    for prefix in ['source_quality','objects','media/real']:
        source=folder/prefix
        if source.exists():shutil.copytree(source,P.OUT/prefix,dirs_exist_ok=True)
    shutil.copy2(folder/'source_candidates.csv',P.OUT/'resume_source_candidates.csv')
    if (folder/'MODEL_REVISIONS.json').exists():shutil.copy2(folder/'MODEL_REVISIONS.json',P.OUT/'MODEL_REVISIONS.json')
    history=P.OUT/'resume_history';history.mkdir(exist_ok=True)
    for name in ['run_status.json','validator_controls/identical_pair.json','study_configuration.json']:
        if (folder/name).exists():shutil.copy2(folder/name,history/name.replace('/','_'))
    for path in (P.OUT/'objects').glob('*.json'):
        record=json.loads(path.read_text())
        if record.get('accepted'):
            record['real_path']=str(P.OUT/'media/real'/(record['artifact_id']+'.png'));P.save(path,record)
    P.save(P.OUT/'preprocessing_migration.json',{'from_version':configuration['version'],'to_version':P.CFG['version'],
        'reused':'Source screening, product masks and normalized images only; source IDs and hashes verified before use',
        'source_records':len(list((P.OUT/'source_quality').glob('*.json'))),
        'accepted_products':sum(json.loads(p.read_text()).get('accepted',False) for p in (P.OUT/'objects').glob('*.json')),
        'reused_generation_or_model_results':False})

def restore():
    if (P.OUT/'study_configuration.json').exists():return
    paths=sorted(Path('/kaggle/input').rglob('RefundProof_VALIDATOR_REPAIR_OUTPUT.zip'))
    if len(paths)>1:raise ValueError('Attach only one corrected resume ZIP')
    if paths:
        import zipfile
        with zipfile.ZipFile(paths[0]) as archive:
            configuration=json.loads(archive.read('RefundProof_PAPER/study_configuration.json'))
            migrate=configuration.get('version')=='RefundProof-validator-repair-3.1'
            if migrate:
                old_worker=archive.read('RefundProof_PAPER/source/correction.py')
                if hashlib.sha256(old_worker).hexdigest()!='b192ed834a2c9df4a929a07e5e35c7b085215f0e34ab340cfc002a51d65f6da7':
                    raise ValueError('Preprocessing migration requires the exact audited v3.1 worker.')
                if any(not entry.is_dir() and any(part in entry.filename.split('/') for part in ['generation','experiments']) for entry in archive.infolist()):
                    raise ValueError('The v3.1 archive contains generation/training work. This migration accepts only the pre-generation failure.')
            elif configuration.get('version')!=P.CFG['version']:
                raise ValueError('Unsupported resume protocol version.')
            for entry in archive.infolist():
                parts=Path(entry.filename).parts
                if not parts or parts[0]!='RefundProof_PAPER' or '..' in parts:raise ValueError('Unexpected corrected archive path')
                relative='/'.join(parts[1:])
                if migrate:
                    if relative=='source_candidates.csv':relative='resume_source_candidates.csv'
                    elif relative in ['run_status.json','validator_controls/identical_pair.json','study_configuration.json']:
                        relative='resume_history/'+relative.replace('/','_')
                    elif not (relative.startswith(('source_quality/','objects/','media/real/')) or relative=='MODEL_REVISIONS.json'):
                        continue
                target=(P.OUT/Path(relative)).resolve()
                if not target.is_relative_to(P.OUT.resolve()):raise ValueError('Unsafe archive path')
                if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(entry) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)
            if migrate:
                for path in (P.OUT/'objects').glob('*.json'):
                    record=json.loads(path.read_text())
                    if record.get('accepted'):
                        record['real_path']=str(P.OUT/'media/real'/(record['artifact_id']+'.png'));P.save(path,record)
                P.save(P.OUT/'preprocessing_migration.json',{'from_version':configuration['version'],'to_version':P.CFG['version'],
                    'reused':'Source screening, product masks and normalized source images only; verified against attached source IDs and hashes before use',
                    'source_records':len(list((P.OUT/'source_quality').glob('*.json'))),
                    'accepted_products':sum(json.loads(p.read_text()).get('accepted',False) for p in (P.OUT/'objects').glob('*.json')),
                    'reused_generation_or_model_results':False})
    else:
        folders=[]
        for configuration in Path('/kaggle/input').rglob('study_configuration.json'):
            try:
                record=json.loads(configuration.read_text())
                if record.get('version') in [P.CFG['version'],'RefundProof-validator-repair-3.1']:folders.append(configuration.parent)
            except (ValueError,OSError):continue
        if len(folders)>1:raise ValueError('Attach only one extracted corrected resume output')
        if folders:
            record=json.loads((folders[0]/'study_configuration.json').read_text())
            if record['version']=='RefundProof-validator-repair-3.1':migrate_extracted_preprocessing(folders[0])
            else:shutil.copytree(folders[0],P.OUT,dirs_exist_ok=True)

ORIGINAL_CHECKPOINT=P.checkpoint
def checkpoint(*args,**kwargs):
    ORIGINAL_CHECKPOINT(*args,**kwargs)
    original=P.ROOT/'RefundProof_PAPER_OUTPUT.zip'
    original.replace(P.ROOT/'RefundProof_VALIDATOR_REPAIR_OUTPUT.zip')

P.prepare_objects=prepare;P.generate=generation;P.setup_core=setup;P.restore=restore;P.checkpoint=checkpoint
try:
    P.main()
    condition_rows=[]
    for path in sorted(P.OUT.glob('final_probabilities_*.csv')):
        scores=pd.read_csv(path).merge(P.PRIMARY[['image_id','source_damage_status']],on='image_id',validate='one_to_one')
        originals=scores[scores.label==0].copy()
        originals['source_condition']=np.where(originals.source_damage_status.str.startswith('caption suggests'),'MLLM_genuine_damage_candidate','MLLM_no_visible_damage_candidate')
        for condition,part in originals.groupby('source_condition'):
            condition_rows.append(dict(model=path.stem.removeprefix('final_probabilities_'),source_condition=condition,n=len(part),
                                       fpr_at_05=float((part.probability>=.5).mean()),human_verified=False))
    pd.DataFrame(condition_rows).to_csv(P.OUT/'source_condition_false_positive_rates.csv',index=False)
    region_files=list((P.OUT/'experiments').glob('*/native_patch_region_metrics.csv'))
    if region_files:pd.concat([pd.read_csv(path) for path in region_files],ignore_index=True).to_csv(P.OUT/'native_patch_region_metrics.csv',index=False)
    external=P.OUT/'external/external_provenance.json'
    if external.exists():
        record=json.loads(external.read_text());record['prior_observation']='This benchmark was evaluated in an earlier run. The current protocol is a repeated external challenge, not an untouched confirmatory test.';P.save(external,record)
    checkpoint('computational_complete_annotation_pending' if not P.FAILURES else 'partial_failed_analyses_annotation_pending',final=True)
except Exception as error:
    print(traceback.format_exc(),flush=True);P.OUT.mkdir(parents=True,exist_ok=True)
    try:
        if P.PRIMARY is not None:P.export_results()
    except Exception:P.save(P.OUT/'collation_failure.json',{'traceback':traceback.format_exc()})
    checkpoint('partial_time_limit' if isinstance(error,P.core.BudgetStop) else 'failed',str(error),final=True)
    sys.exit(2)
