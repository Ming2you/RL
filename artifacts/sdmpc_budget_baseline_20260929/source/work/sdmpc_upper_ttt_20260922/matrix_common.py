"""Specs 05/08: historical environment and enforced single logical CPU."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HIST=ROOT/'work/sdmpc_matrix_14400_20260912/historical_tree'
SCENARIOS=('sweet_190_w','sweet_190_skew15_w','sweet_190_incident_w','sweet_220_w','sweet_220_skew15_w','sweet_220_incident_w')
MODES={'NC':'NO-CONTROL','STACKELBERG':'P-STACK-WU-FAITHFUL-ALLPRICE-JOINT',
       'PFO':'WU-FAITHFUL-FOLLOWER','PFO_SPLIT':'WU-FAITHFUL-FOLLOWER'}
PROTOCOLS=ROOT/'outputs/sdmpc_upper_ttt_20260922/protocols_0'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p); temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); temp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(mask=None):
    if os.name!='nt': raise RuntimeError('Windows affinity contract required')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    k.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    proc=k.GetCurrentProcess(); before,system=ctypes.c_size_t(),ctypes.c_size_t()
    if not k.GetProcessAffinityMask(proc,ctypes.byref(before),ctypes.byref(system)): raise ctypes.WinError(ctypes.get_last_error())
    selected=int(mask) if mask is not None else before.value & -before.value
    if selected<=0 or selected & (selected-1) or not selected & before.value: raise RuntimeError('Unavailable single CPU mask')
    if not k.SetProcessAffinityMask(proc,selected): raise ctypes.WinError(ctypes.get_last_error())
    after=ctypes.c_size_t()
    if not k.GetProcessAffinityMask(proc,ctypes.byref(after),ctypes.byref(system)): raise ctypes.WinError(ctypes.get_last_error())
    if after.value!=selected: raise RuntimeError('Affinity verification failed')
    return dict(pid=os.getpid(),before_mask=before.value,mask=after.value,logical_cpu_index=selected.bit_length()-1,system_mask=system.value)
def environment(mode):
    spec=read(ROOT/'work/sdmpc_matrix_14400_20260912/historical_environment.json')
    for key in spec['controller_environment_keys']+spec['launch_unset']: os.environ.pop(key,None)
    os.environ.update(spec['environment'])
    if mode=='CENTRALIZED_SLSQP':raise ValueError('Centralized SLSQP excluded by user from this extension')
    os.environ.pop('CENT_SLSQP',None)
    os.environ.pop('CENT_SLSQP_MAXEVAL',None)  # Preserve native configured default: no evaluation cap.
    if mode in ('PFO_SPLIT','STACKELBERG','SDMPC'): os.environ['PFO_SPLIT']='2'
    else: os.environ.pop('PFO_SPLIT',None)  # bool('0') would incorrectly enable it.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'NUMEXPR_MAX_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['OMP_DYNAMIC']='FALSE'; os.environ['MKL_DYNAMIC']='FALSE'
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONPATH']=os.pathsep.join((str(ROOT/'work/sdmpc_20260911/deps'),str(HIST)))
    return {k:os.environ.get(k) for k in set(spec['controller_environment_keys']+list(spec['environment']))|
        {'OMP_DYNAMIC','MKL_DYNAMIC','NUMBA_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_MAX_THREADS'}}
