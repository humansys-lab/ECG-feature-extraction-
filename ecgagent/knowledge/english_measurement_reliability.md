# ECG Measurement Reliability Reference

This reference describes how to weight ecgfeat observations; it is not patient evidence.

## Evidence hierarchy

Prefer reportable global values and stable repeated observations. Next use independently corroborated cross-lead or cross-beat morphology. Treat detector summaries and candidate-event streams as prompts for investigation. Unavailable, ambiguous or unvalidated observations cannot establish a diagnosis.

## Representative beats and morphology groups

A representative beat describes one morphology family, not necessarily the entire recording. Review group prevalence, member count, longest run, template stability and outliers before applying its morphology to the record. A conclusion driven by one beat should be reproduced in another beat or downgraded.

## P-wave boundaries

An accepted P-wave extraction can still have T/A ambiguity. Use onset and offset confidence, valid-lead support, repeated clean boundaries and stable morphology clusters. Candidate atrial-event rows from one stream are not independent observations even when multiple source leads are listed.

## ST-T-U and interval endpoints

Interpret ST changes only when the measurement is reliable and forms a coherent contiguous-lead distribution. For T waves, review polarity, amplitude, symmetry, T-wave signal quality and opposing leads. QT/QTc requires a reliable T-wave endpoint; a fallback endpoint is an extraction path, not proof of a flat, fused or abnormal T wave.

## Pacing and native beats

Spike detection, spike-to-QRS matching, paced-beat flags and pacing-derived morphology can belong to one algorithmic chain. Confirm pacing with non-conflicted marker timing plus compatible repeated morphology. When pacing is suspected, use native non-paced beats for infarct and repolarization review whenever available.

## Reliability language

A reliability flag lowers evidence weight but does not erase a measured value. State the limitation in the same claim, seek independent corroboration and lower confidence when the conclusion materially depends on that measurement.
