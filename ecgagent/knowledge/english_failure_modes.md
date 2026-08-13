# ECG Agent Failure Modes

This reference lists common reasoning failures that should trigger targeted falsification.

- Do not infer a rhythm mechanism from rate alone.
- Do not convert residual-signal cycle length into atrial rate.
- Do not convert atrial candidate-event counts into AV conduction ratios.
- Do not treat repeated fields from one detector chain as independent corroboration.
- Do not infer conduction delay from QRS notching or fragmentation without prolonged QRS duration.
- Do not diagnose pre-excitation from delta candidates without independent PR shortening.
- Do not diagnose flutter from a fixed ventricular rate without organized cross-lead atrial activity.
- Do not diagnose atrial fibrillation from one fibrillatory-wave candidate without rhythm and P-wave evidence.
- Do not apply chamber-voltage criteria when pacing or substantial QRS widening invalidates them.
- Do not infer ischemic etiology from a single low-quality ST/T observation.
- Do not equate unavailable PR with absent P waves or unavailable QT with uninformative T waves.
- Do not generalize one representative beat to the whole recording without morphology-group support.
- Do not use a technical quality flag to prove a named lead reversal unless its direct criteria pass.
- Do not turn a missing value into a normal or abnormal finding.
