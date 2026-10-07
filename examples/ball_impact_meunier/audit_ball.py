import importlib.util, json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
case = root / 'simulation/ball_impact_meunier'
spec = importlib.util.spec_from_file_location('ball', case / 'postprocess_ball.py')
ball = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ball)
results = {}
for prefix in ('BALL_IMPACT', 'BALL_FINE'):
    path = case / (prefix + 'T01.csv')
    if not path.exists():
        continue
    history = ball.read_history(path)
    results[prefix] = ball.history_metrics(history)
    reference_by_id = ball.read_reference_nodes(case / (prefix + '_0000.rad'))
    envelope = dict(min_J=float('inf'), max_J=0., min_I1bar=float('inf'),
                    max_I1bar=0., min_I2bar=float('inf'), max_I2bar=0.,
                    maximum_von_mises_MPa=0., frames=0)
    for vtk in sorted((case / ('vtk_' + prefix)).glob('*.vtk')):
        t, points, cells, vm = ball.read_vtk(vtk)
        tokens = vtk.read_text().split()
        start = tokens.index('NODE_ID') + 5
        ids = np.asarray(tokens[start:start+len(points)], dtype=int)
        ref = np.asarray([reference_by_id.get(int(i), (np.nan,)*3) for i in ids])
        conn = np.asarray(cells)
        x, X = points[conn], ref[conn]
        current = np.transpose(x[:,1:] - x[:,:1], (0,2,1))
        original = np.transpose(X[:,1:] - X[:,:1], (0,2,1))
        F = current @ np.linalg.inv(original)
        J = np.linalg.det(F)
        if np.any(J <= 0):
            raise ValueError('Non-positive element Jacobian')
        eig = np.linalg.eigvalsh(np.transpose(F,(0,2,1)) @ F) / J[:,None]**(2/3)
        I1 = eig.sum(axis=1)
        I2 = eig[:,0]*eig[:,1] + eig[:,1]*eig[:,2] + eig[:,2]*eig[:,0]
        for key, value in [('min_J',J.min()),('max_J',J.max()),
                           ('min_I1bar',I1.min()),('max_I1bar',I1.max()),
                           ('min_I2bar',I2.min()),('max_I2bar',I2.max()),
                           ('maximum_von_mises_MPa',vm.max())]:
            envelope[key] = float(min(envelope[key],value) if key.startswith('min_')
                                  else max(envelope[key],value))
        envelope['frames'] += 1
    results[prefix]['animation_envelope'] = envelope
(case / 'ball_impact_qa.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
