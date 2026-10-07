import json, os, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
case = root / 'examples/figure10_cann_csr'
nn_dir = root / 'src/nn_invariant'
sys.path.insert(0, str(nn_dir))
from solver.network_invariant import FFBPNetworkInvariant
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

source = (case / 'benchmark_evaluation_cost.py').read_text()
start = source.index('class FFBPNetworkInvariant')
end = source.index('torch.set_num_threads', start)
source = source[:start] + source[end:]
source = source.replace('N_SCALAR = 1_000_000', 'N_SCALAR = 20_000')
source = source.replace('(CASE / "evaluation_cost_meunier.json").write_text', '(root / "benchmarks/material_point/material_point_rerun.json").write_text')
env = dict(globals(), __file__=str(case / 'benchmark_evaluation_cost.py'))
print('BLAS pools before limiting:', threadpool_info(), flush=True)
with threadpool_limits(limits=1):
    exec(compile(source, 'audited_benchmark', 'exec'), env)
    z = env['inv_np'] @ env['Wh'].T + env['bh']
    ref = env['nn_autograd'](env['inv_t']).numpy()
    a, b = env['nn_numpy'](env['I1'], env['I2'])
    err = float(np.max(np.abs(ref - np.column_stack((a,b)))))
    print('all-million maximum derivative error:', err)
    print('preactivation min/max:', float(z.min()), float(z.max()))
    print('torch threads:', torch.get_num_threads())
    print('BLAS pools during benchmark:', threadpool_info())
    report = json.loads((root / 'benchmarks/material_point/material_point_rerun.json').read_text())
    report['maximum_absolute_derivative_error_all_samples'] = err
    report['preactivation_range'] = [float(z.min()), float(z.max())]
    report['original_network_source'] = str(nn_dir / 'solver/network_invariant.py')
    report['blas_pools'] = threadpool_info()
    (root / 'benchmarks/material_point/material_point_rerun.json').write_text(json.dumps(report, indent=2))
