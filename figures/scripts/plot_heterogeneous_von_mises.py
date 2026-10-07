"""Compare exported solid von Mises stress at the selected ~20 N states."""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation/figure10_cann_csr"))
from postprocess import read_legacy_vtk, read_msh2_nodes, front_surface, boundary_segments

def load(folder, mesh):
    path = ROOT / "simulation" / folder
    points, cells, ids = read_legacy_vtk(path / "selected_state.vtk")
    nodes = read_msh2_nodes(path / mesh)
    original = np.array([nodes.get(int(i), [np.nan]*3) for i in ids])
    faces, owners = front_surface(cells, original)
    tokens = (path / "selected_state.vtk").read_text(encoding="ascii").split()
    start = tokens.index("3DELEM_Von_Mises") + 5
    stress = np.array(tokens[start:start+len(cells)], dtype=float)
    start = tokens.index("3DELEM_Stress") + 2
    tensor = np.array(tokens[start:start+9*len(cells)], dtype=float).reshape(-1,3,3)
    dev = tensor - np.trace(tensor,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
    calculated = np.sqrt(1.5*np.sum(dev*dev,axis=(1,2)))
    assert np.allclose(stress[owners], calculated[owners], rtol=2e-4, atol=1e-6)
    assert np.isfinite(stress[owners]).all() and np.min(stress[owners]) >= 0
    state = json.loads((path/"selected_state.json").read_text())
    return points, faces, stress[owners], state

cases = [load("figure10_cann_csr", "figure10.msh"), load("figure10_cann_csr_fine", "figure10_fine.msh")]
plt.rcParams.update({"font.size":9, "font.family":"DejaVu Sans"})
fig, axes = plt.subplots(1,2,figsize=(6.2,5.3))
norm = Normalize(0, 2.6, clip=True)
for ax, (points, faces, values, state), label in zip(axes,cases,["(a) Coarse mesh","(b) Refined mesh"]):
    coll = PolyCollection([points[list(f),:2] for f in faces], array=values, cmap="viridis", norm=norm, edgecolors="none", rasterized=True)
    ax.add_collection(coll)
    ax.add_collection(LineCollection(boundary_segments(faces,points),colors="0.2",linewidths=.4))
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(label,fontsize=10)
    print(label, "reaction_N=", state["moving_grip_reaction_N"], "surface_vm_range_MPa=", (values.min(),values.max()))
fig.subplots_adjust(left=.03,right=.84,bottom=.03,top=.92,wspace=.12)
cax=fig.add_axes([.87,.15,.025,.65])
cb = fig.colorbar(coll, cax=cax, ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.6])
cb.set_label("von Mises stress (MPa)", fontsize=6.5)
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.outline.set_linewidth(0.5)
out=ROOT/"figs/heterogeneous_von_mises_mesh_comparison"
fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight")
fig.savefig(out.with_suffix(".png"),dpi=300,bbox_inches="tight")
