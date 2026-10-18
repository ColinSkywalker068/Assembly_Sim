"""Run two explicitly selected GPUs and at most two CPU geometry jobs."""
import argparse,json,os,subprocess,sys,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--gpu-primary',required=True);p.add_argument('--gpu-secondary',required=True)
    a=p.parse_args();b=a.base;out=b/'skull-ablation-v2-run';out.mkdir(exist_ok=True)
    status={};lock=threading.Lock();warm_ready=threading.Event();warm_ok=[];screen_jobs=[]
    module='tasks.dual_arm_crag_skull.experiments.skull_ablation_v2'
    dataset=b/'skull-ablation-v2-data'
    if not (dataset/'cohort.json').exists():raise ValueError('freeze cohort first')
    if (out/'status.json').exists():raise ValueError('run exists; do not overwrite experiment')
    def job(name,mode,arm,checkpoint=None,gpu=None):
        folder=out/arm;folder.mkdir(exist_ok=True)
        command=[sys.executable,'-u','-m',module,mode,'--dataset',str(dataset),'--output',str(folder),'--arm',arm]
        if mode=='run':command+=['--repository',str(b/'GARF'),'--checkpoint',str(checkpoint)]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with lock:
            status[name]={'state':'running','gpu':gpu};(out/'status.json').write_text(json.dumps(status,indent=2))
        with open(out/(name+'.log'),'w') as log:
            result=subprocess.run(command,cwd=b/'Assembly_Sim',env=env,stdout=log,stderr=subprocess.STDOUT)
        with lock:
            status[name].update(state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode)
            (out/'status.json').write_text(json.dumps(status,indent=2))
        if result.returncode:raise RuntimeError(name+' failed; inspect log')
    with ThreadPoolExecutor(max_workers=2) as screening:
        def arm(name,gpu,checkpoint):
            job(name,'run',name,checkpoint,gpu)
            with lock:screen_jobs.append(screening.submit(job,name+'-screen','screen',name))
        def primary():
            try:
                job('warmup','run','warmup',b/'GARF.ckpt',a.gpu_primary);warm_ok.append(True)
            finally:warm_ready.set()
            arm('flow20',a.gpu_primary,out/'warmup/encoder.ckpt')
            arm('multi200',a.gpu_primary,out/'warmup/encoder.ckpt')
        def secondary():
            arm('previous',a.gpu_secondary,b/'skull-finetune-v1/joint.ckpt')
            warm_ready.wait()
            if not warm_ok:raise RuntimeError('shared warmup failed')
            arm('flow200',a.gpu_secondary,out/'warmup/encoder.ckpt')
        with ThreadPoolExecutor(max_workers=2) as gpu_jobs:
            futures=[gpu_jobs.submit(primary),gpu_jobs.submit(secondary)]
            for f in futures:f.result()
        for f in screen_jobs:f.result()


if __name__=='__main__':main()
