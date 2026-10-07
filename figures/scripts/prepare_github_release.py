"""Prepare a curated, local release candidate; run in WSL with Python 3."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'release' / 'hyperelasticity-symbolic-discovery'
NN = Path('/home/guanjs/NN-constitutive/FFNN/hyperelasticity/NN_invariant')
RAD = Path('/home/guanjs/OpenRadioss-latest-20251211')
manifest = []

def copy(source, target):
    target = DEST / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append({'path': str(target.relative_to(DEST)), 'bytes': target.stat().st_size,
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                     'source': str(source)})

def tree(source, target, extensions):
    for path in sorted(source.rglob('*')):
        if not path.is_file() or any(p in {'__pycache__','.venv','.git','output','vendor'}
                                     for p in path.relative_to(source).parts):
            continue
        if path.suffix in extensions:
            copy(path, Path(target) / path.relative_to(source))

DEST.mkdir(parents=True, exist_ok=True)
for name in ('main.py','activation_function.py','run_train.sh','run_symbolic.sh','run_all.sh','README.md'):
    copy(NN/name, Path('src/nn_invariant')/name)
for name in ('cli','preprocess','solver','postprocess','Symbolic_Regression','synthetic_data'):
    tree(NN/name, Path('src/nn_invariant')/name, {'.py','.sh','.md','.jl'})
for name in ('export_openradioss.py','requirements.txt','README.md'):
    copy(RAD/'tools/nn_invariant'/name, Path('integrations/openradioss')/name)
tree(RAD/'tools/nn_invariant/examples','integrations/openradioss/examples',{'.json','.md'})
for name in ('lecmuser01.f90','luser01.f90','build.sh','README.md'):
    copy(RAD/'tools/userlib/examples/law291_userlib'/name,
         Path('integrations/openradioss/userlib')/name)
copy(RAD/'LICENSE.md','LICENSES/OpenRadioss-AGPL-3.0.md')
tree(ROOT/'scripts','figures/scripts',{'.py'})
tree(ROOT/'figs/data','results/figure_data',{'.csv','.json'})
for folder in ('figure10_cann_csr','ball_impact_meunier'):
    source = ROOT/'simulation'/folder
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {'.py','.sh','.md','.geo','.msh','.rad','.flat','.csv','.json'}:
            if path.name.startswith('evaluation_cost') and folder == 'figure10_cann_csr':
                continue
            copy(path, Path('examples')/folder/path.name)
source = ROOT/'simulation/figure10_cann_csr/nn_cache/output'
for folder in ('NN_output','SR_output'):
    for name in ('ffbp_model.pt','ffbp_model.json','material_package.json','simplification_summary.json',
                 'experiment_predictions.csv','experiment_prediction_summary.json'):
        path = source/folder/name
        if path.exists(): copy(path, Path('models/meunier')/folder/name)
for name in ('audit_material_point.py','material_point_audit.json'):
    copy(ROOT/'tmp'/name,Path('benchmarks/material_point')/name)
copy(ROOT/'simulation/brain_data/README.md','data/brain-source.md')
# Local inventory includes source paths for audit; public inventory contains no local paths.
(DEST/'MANIFEST.json').write_text(json.dumps(
    [{k:v for k,v in m.items() if k!='source'} for m in manifest],indent=2)+'\n')
(ROOT/'release/source-inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Prepared {len(manifest)} files, {sum(m["bytes"] for m in manifest)/1e6:.2f} MB in {DEST}')
