"""진행 중 실험의 파일을 수정하지 않고 역사 런타임을 별도 프로세스로 읽는다."""
from pathlib import Path
import sys
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/sdmpc_engine_speed_20260912'))
from profile_current import bootstrap,read,write,restore_state


def load(step=30):
    cell=ROOT/'outputs/sdmpc_frozen_price_20260912/matrix_C_attempt_0/interval_vsl/sweet_190_w'
    rc,_=bootstrap(cell)
    source=ROOT/'outputs/extended_matrix_no_slsqp_20260920/attempt_1/SDMPC6/sweet_190_skew15_w'
    data=read(source/f'decision_{step:03d}/input.json')
    cfg=rc.restore_historical_config(read(source/'factory_config.json'))
    opts=replace(rc.PlayerSDMPCOptions(**read(source/'solver_options.json')),max_iterations=10)
    demand=[]
    for d in data['forecast']:
        d=dict(d);d['freeway_lane_loss']={link:{int(i):v for i,v in values.items()} for link,values in d.get('freeway_lane_loss',{}).items()}
        demand.append(rc.DemandStep(**d))
    sys.path[:0]=[str(HERE),str(ROOT/'work/sdmpc_block_speed_ordered_20260919'),str(ROOT/'work/sdmpc_sparse_local_20260916')]
    return rc,cfg,opts,restore_state(rc,data['state']),demand,rc.ControlAction(**data['previous']),rc.ControlAction(**data['warm'])
