# Hyperelasticity Symbolic Discovery

Source and data for neural-surrogate-assisted discovery of compact
hyperelastic strain-energy laws and deployment in OpenRadioss.

This initial publication preserves selected discovery sources, numerical
datasets and frozen results. The workflows retain some historical paths and
are not yet certified as a fully portable, end-to-end reproducibility release.

## Contents

- `src/nn_invariant`: neural training, symbolic regression and pruning sources.
- `integrations/openradioss`: exporter and dynamic USER01 material library.
- `models/meunier`: trained network and deployed symbolic material package.
- `examples`: perforated-plate and Meunier ball-impact decks, meshes and results.
- `benchmarks/material_point`: Intel single-thread audit supporting the SI table.
- `results/figure_data`, `figures/scripts`: figure inputs and plotting/collection scripts.
- `MANIFEST.json`: file sizes and SHA256 checksums of curated source assets.

## Quick start (Linux / WSL)

Create a Python environment and install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/nn_invariant/main.py --help
python integrations/openradioss/export_openradioss.py export-openradioss \
  --config models/meunier/SR_output/material_package.json \
  --output /tmp/meunier_material.flat --nu 0.495
```

For OpenRadioss, set `OPENRADIOSS_ROOT` to the separately installed source/build
tree, then build the user library with
`bash integrations/openradioss/userlib/build.sh`.
Set the case launcher's `USERLIB` to the generated library if running the
historical launch scripts outside the original solver layout.

The scalar/vectorized single-thread CPU benchmark can be rerun with:

```bash
python benchmarks/material_point/audit_material_point.py
```

It writes `material_point_rerun.json` alongside the original audit results.
Timing ratios depend on hardware and Python numerical backends.
PySR initializes a Julia environment; stochastic rediscovery does not
guarantee an identical expression to the frozen model.

OpenRadioss is obtained separately from https://github.com/OpenRadioss/OpenRadioss.
Its license is preserved under `LICENSES/`. This does not assign the same
license to every author-owned file; licensing for those files is pending.

## Remaining release work

1. Replace author-specific absolute paths and reconcile the exporter vendored
   implementation with `src/nn_invariant`.
2. Pin dependency and solver versions; add portable run/build instructions.
3. Select source licenses and document dataset redistribution conditions.
4. Verify clean-environment discovery, package export and solver smoke runs.
5. Publish a validated tagged release and archive it with a persistent identifier.

Do not describe this snapshot as a complete reproducibility release until
these checks are completed. Experimental source-workbooks and article scans
are not included in this snapshot.
