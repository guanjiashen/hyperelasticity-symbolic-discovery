# Initial publication checks (2026-10-08)

Performed against the curated source layout using the existing WSL Python
environment (not a newly installed clean environment):

- All 59 Python source files parsed successfully.
- Neural-training `main.py --help` imported and completed successfully.
- Exporting the frozen Meunier material package produced the expected
  six-term polynomial (five nonconstant terms), reference shear modulus
  0.3332051983 MPa and bulk modulus 33.20945143 MPa.
- The relocated single-thread material-point audit completed successfully,
  including million-point derivative comparisons and 20,000 pointwise calls.

The underlying coarse and refined Meunier ball-impact analyses terminated
normally before preparation of this publication. Their frozen history and
QA tables are included; the entire simulation was not rerun from a clean
checkout for this upload.

Dependency pinning, clean-environment full discovery, historical plotting
path adaptation and complete solver workflows remain to be validated.
No DOI or validated version release is claimed by this source/data upload.
