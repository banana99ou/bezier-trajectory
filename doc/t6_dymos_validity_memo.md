# T6 Dymos Validity Memo

- comparison valid: `False`
- framework gate: `True`
- physics gate: `True`
- fairness gate: `True`
- objective recompute gate: `False`
- dense KOZ gate: `True`
- warm export gate: `True`
- naive init gate: `True`
- orbital credibility gate: `False`

## Run Summary

- naive success: `True`, iterations: `4`, runtime: `1.324 s`, objective: `0.147371`
- warm success: `True`, iterations: `4`, runtime: `1.322 s`, objective: `0.199106`

## Interpretation Boundary

- Treat the comparison as trustworthy only if all validity gates pass.
- If any gate fails, the run is an engineering diagnostic and not paper-facing T6 evidence.
