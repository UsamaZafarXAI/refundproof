"""Per-file downloads with network retries and progress reporting."""
import os,time,re,runpy,json,threading,http.client,fnmatch
from pathlib import Path,PurePosixPath
from email.utils import parsedate_to_datetime
from datetime import datetime,timezone
os.environ['HF_HUB_DISABLE_XET']='1'
import requests
from urllib3.exceptions import ProtocolError,ReadTimeoutError
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
import huggingface_hub
NETWORK_ERRORS=(requests.exceptions.ConnectionError,requests.exceptions.ChunkedEncodingError,requests.exceptions.Timeout,ProtocolError,ReadTimeoutError,http.client.IncompleteRead)
BUDGET_HOOK=None
def retry_delay(response):
    headers=response.headers if response is not None else {}
    value=headers.get('Retry-After','')
    try:return max(1,float(value))
    except (ValueError,TypeError):
        if value:
            try:return max(1,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (ValueError,TypeError):pass
    match=re.search(r't\s*=\s*(\d+)',headers.get('RateLimit',''))
    return int(match.group(1))+2 if match else 302
def retry(operation,label):
    for attempt in range(8):
        if BUDGET_HOOK is not None:BUDGET_HOOK()
        try:return operation()
        except (HfHubHTTPError,)+NETWORK_ERRORS as exc:
            response=getattr(exc,'response',None);status=getattr(response,'status_code',None)
            if isinstance(exc,HfHubHTTPError) and status not in [408,429,500,502,503,504]:raise
            if attempt==7:raise
            delay=retry_delay(response) if status==429 else min(60,5*2**attempt)
            print(f'Download interruption ({type(exc).__name__}, HTTP {status or "n/a"}): {label}. Retrying in {delay:.0f}s [{attempt+1}/7]. Completed files are kept.',flush=True)
            while delay>0:
                if BUDGET_HOOK is not None:BUDGET_HOOK()
                step=min(30,delay);time.sleep(step);delay-=step
                if delay>0:print(f'Download retry wait: {delay:.0f}s remaining.',flush=True)
def match_patterns(name,patterns):
    if patterns is None:return True
    if isinstance(patterns,str):patterns=[patterns]
    return any(fnmatch.fnmatch(name,p) for p in patterns)
def save_provenance():
    if not os.environ.get('WORK_DIR'):return
    out=Path(os.environ['WORK_DIR'])/'results';(out/'source').mkdir(parents=True,exist_ok=True)
    (out/'source'/'download_transport.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    (out/'download_settings.json').write_text(json.dumps({'authenticated':True,'token_saved_to_output':False,'max_workers':1,'xet_disabled':True,'per_file_retry_attempts':7,'progress_every_files':25,'heartbeat_seconds':30,'retry_policy':'Rate-limit reset, network drops and transient HTTP errors; SDK partial-file resume'},indent=2),encoding='utf-8')
def download(repo_id,**kwargs):
    save_provenance()
    token=os.environ['HF_TOKEN'];repo_type=kwargs.get('repo_type','model');api=HfApi(token=token)
    print(f'Checking file inventory: {repo_id}',flush=True)
    info=retry(lambda:api.repo_info(repo_id,repo_type=repo_type,revision=kwargs.get('revision'),files_metadata=True),repo_id+' inventory')
    files=[f for f in info.siblings if match_patterns(f.rfilename,kwargs.get('allow_patterns')) and not (kwargs.get('ignore_patterns') is not None and match_patterns(f.rfilename,kwargs['ignore_patterns']))]
    files.sort(key=lambda f:f.rfilename)
    if not files:raise ValueError('No matching files for '+repo_id)
    total=sum(f.size or 0 for f in files);started=time.monotonic();done=0;completed_bytes=0
    root=Path(kwargs['local_dir']) if kwargs.get('local_dir') else None
    print(f'{repo_id}: {len(files):,} files, {total/1024**3:.2f} GiB. Completed local files are reused.',flush=True)
    for f in files:
        parts=PurePosixPath(f.rfilename).parts
        if '..' in parts or PurePosixPath(f.rfilename).is_absolute():raise ValueError('Unsafe repository filename')
        if done%25==0 or (f.size or 0)>50*1024**2:print(f'Downloading {done+1:,}/{len(files):,}: {f.rfilename}',flush=True)
        call=dict(repo_id=repo_id,filename=f.rfilename,repo_type=repo_type,revision=info.sha,token=token)
        for key in ['local_dir','cache_dir','force_download','local_files_only']:
            if key in kwargs:call[key]=kwargs[key]
        path=retry(lambda:hf_hub_download(**call),f.rfilename)
        if root is None:root=Path(path).parents[len(parts)-1]
        done+=1;completed_bytes+=f.size or Path(path).stat().st_size
        if done%25==0 or done==len(files):print(f'Download progress {repo_id}: {done:,}/{len(files):,} files ({100*done/len(files):.1f}%), {completed_bytes/1024**3:.2f}/{total/1024**3:.2f} GiB, elapsed {(time.monotonic()-started)/60:.1f} min.',flush=True)
    return str(root)
def heartbeat(stop,out):
    while not stop.wait(30):
        log=out/'run.log';last='Preparing or downloading dataset/models'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-2048));lines=f.read().decode('utf-8',errors='replace').splitlines()
            if lines:last=lines[-1][:220]
        completed=sum(1 for _ in (out/'experiments').glob('*/done.json'))
        print(f'[{time.strftime("%H:%M:%S")}] Still running. Completed experiments: {completed}. Latest pipeline message: {last}',flush=True)
if __name__=='__main__':
    token=os.environ.get('HF_TOKEN','').strip()
    if not token:raise RuntimeError('Enter a free HF Read token. No compute credits are needed.')
    retry(lambda:HfApi(token=token).whoami(),'Hugging Face login')
    huggingface_hub.snapshot_download=download
    print('HF login verified. Connection retries, per-file caching and visible progress enabled.',flush=True)
    out=Path(os.environ['WORK_DIR'])/'results';(out/'source').mkdir(parents=True,exist_ok=True)
    (out/'source'/'download_transport.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    (out/'download_settings.json').write_text(json.dumps({'authenticated':True,'token_saved_to_output':False,'max_workers':1,'xet_disabled':True,'per_file_retry_attempts':7,'progress_every_files':25,'heartbeat_seconds':30,'retry_policy':'Rate-limit reset, network drops and transient HTTP errors; SDK partial-file resume'},indent=2),encoding='utf-8')
    stop=threading.Event();thread=threading.Thread(target=heartbeat,args=(stop,out),daemon=True);thread.start()
    try:runpy.run_path(str(Path(__file__).with_name('runner.py')),run_name='__main__')
    finally:stop.set();thread.join(timeout=2)
