# Limitations and intended use

> This software is research and engineering software. It is not a medical
> device and is not intended to diagnose, treat, cure, or prevent disease.
> Outputs require independent validation for the intended use and must not be
> used as a substitute for professional medical judgment.

Records carry `intended_use = "research_and_engineering_only"`; no record is
medically cleared, approved or clinically validated.

- All published fields are `unvalidated` in schema 1.0.0.
- P-wave delineation is the weakest endpoint in the repository benchmarks,
  especially at short RR intervals or when the P wave overlaps the previous T wave.
- LUDB distributes amplitude-normalized signals, so LUDB cannot validate amplitudes.
- The four-class ST morphology was never validated and is anti-correlated with
  ischemia; it is not published and must not be used as ischemia evidence.
- `ecginterpret` statements are rule outputs for research; they are not diagnoses.
- The pipeline is inspired by publicly documented practice; it is not a
  reproduction of any proprietary product.
