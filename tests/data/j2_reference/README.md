# J2 Reference Data

`egm2008_degree2_samples.json` is a normalized offline fixture for the J2 validation suite.

Source:
- raw coefficient table: `EGM-08norm100.txt` from the public CelesTrak/Vallado astrodynamics repository
- extracted term: fully normalized `C20`

Construction:
- deterministic sample positions are generated at fixed altitude / latitude / longitude combinations
- the reference gravity field is computed from a scalar potential using the normalized `C20` term
- the reference acceleration is then obtained by central differencing the potential

This is intentionally separate from the production closed-form J2 acceleration in the optimizer so the
regression suite has an independent comparison path.

Refresh intentionally with:

```bash
python tools/fetch_j2_reference_data.py --output tests/data/j2_reference/egm2008_degree2_samples.json
```
