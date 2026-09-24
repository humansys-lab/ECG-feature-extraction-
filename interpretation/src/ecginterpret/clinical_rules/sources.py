RULESET_VERSION = "2026.07.28"
SCHEMA_VERSION = "clinical_rules.v2"
RESOLVER_POLICY_VERSION = "coverage-confidence-v1"

# ---------------------------------------------------------------------------
# Declared diagnostic baseline
# ---------------------------------------------------------------------------
# `clinical_rules` is the layer that emits the final diagnostic statements, so
# it needs exactly one standards family to arbitrate numeric conflicts. That
# family is the AHA/ACCF/HRS 2009 ECG standardization series plus the Fourth
# Universal Definition of Myocardial Infarction (2018).
#
# The other two engines in this package stay on their own vendor sources on
# purpose -- interpret.py on the Philips DXL threshold reference, glasgow_rules
# on the Glasgow GAN Physician's Guide. Neither arbitrates the other. When a
# consumer needs "the" diagnosis, it must read clinical_rules.
BASELINE_STANDARD = {
    "family": "AHA/ACCF/HRS ECG standardization (2009) + UDMI 4th edition (2018)",
    "authoritative_layer": "feature_extraction.ecgfeat.clinical_rules",
    "version": RULESET_VERSION,
    "non_arbitrating_layers": [
        "feature_extraction.ecgfeat.interpret (Philips DXL threshold reference)",
        "feature_extraction.ecgfeat.glasgow_rules (Glasgow GAN Physician's Guide)",
    ],
}

# Conflicts between the baseline and the textbook conventions catalogued in
# docs/心电图诊断规则系统综述.md, and how each was resolved. Recorded here so a
# reviewer can see that a divergence is a decision rather than an oversight.
BASELINE_DIVERGENCES = [
    {
        "topic": "pathological Q-wave duration",
        "baseline": ">=30 ms (>=20 ms in V2-V3)",
        "textbook_alternative": ">=40 ms",
        "resolution": "baseline",
        "rationale": (
            "UDMI 2018 lowered the general limb/precordial cutoff to 30 ms. "
            "Kept, because the whole ischemia domain cites UDMI. The 40 ms "
            "convention remains reachable by raising "
            "IschemiaThresholds.pathological_q_duration_ms."
        ),
    },
    {
        "topic": "QTc correction formula",
        "baseline": "Hodges primary; Bazett retained for the >500 ms alert",
        "textbook_alternative": "Bazett primary",
        "resolution": "baseline",
        "rationale": (
            "Bazett over-corrects at tachycardia and produced a false "
            "prolonged-QT call on JS00059 (see "
            "tests/test_clinical_intervals.py). All four corrections are "
            "still reported in evidence."
        ),
    },
    {
        "topic": "definite QT prolongation tier",
        "baseline": "prolonged at 450 ms (male) / 460 ms (female)",
        "textbook_alternative": ">=480 ms is 'definite' prolongation",
        "resolution": "adopted_alternative_as_additional_tier",
        "rationale": (
            "The 450/460 entry threshold is kept, and a distinct >=480 ms "
            "markedly-prolonged tier was added so the escalation step between "
            "'prolonged' and the >500 ms alert is no longer missing."
        ),
    },
    {
        "topic": "left posterior fascicular block axis floor",
        "baseline": "+90 deg",
        "textbook_alternative": "+110 deg",
        "resolution": "adopted_alternative",
        "rationale": (
            "LPFB is a diagnosis of exclusion and the +90 deg floor made it "
            "fire on ordinary vertical hearts. Raised to +110 deg, which is "
            "also what AHA/ACCF/HRS Part III recommends."
        ),
    },
    {
        "topic": "left anterior fascicular block axis window",
        "baseline": "-45 to -90 deg",
        "textbook_alternative": "<= -30 deg accepted by some texts",
        "resolution": "adopted_alternative_as_additional_tier",
        "rationale": (
            "The -45 deg window matched nothing in a 100-record cohort while "
            "a record labelled 'axis left shift' sat at -41 deg. The -30 to "
            "-45 deg band now yields a probable_lafb_pattern rather than "
            "widening the definite window, because that band is where LVH, "
            "horizontal heart and prior inferior infarction cluster. Made "
            "safe by also implementing the S(III)>S(II) and R(aVL)>R(I) "
            "vector checks, which the rule previously omitted."
        ),
    },
    {
        "topic": "sinus bradycardia cutoff",
        "baseline": "60 bpm",
        "textbook_alternative": "50-60 bpm",
        "resolution": "baseline",
        "rationale": (
            "Single source of truth is now "
            "IntervalThresholds.adult_bradycardia_bpm; basic_rhythm no longer "
            "hardcodes it. interpret.py keeps DXL's 50 bpm under its own "
            "citation."
        ),
    },
    {
        "topic": "sinus P-axis window",
        "baseline": "0 to +75 deg",
        "textbook_alternative": "upright II/III/aVF with inverted aVR (~0 to +90 deg)",
        "resolution": "baseline",
        "rationale": (
            "Now expressed once as "
            "IntervalThresholds.sinus_p_axis_min_deg/max_deg instead of a "
            "literal in basic_rhythm."
        ),
    },
    {
        "topic": "prominent U wave amplitude",
        "baseline": "not quantified",
        "textbook_alternative": ">1-2 mm, or >25 % of the T wave in the lead",
        "resolution": "adopted_alternative",
        "rationale": (
            "Taken from LITFL/AHA Part IV. The lower end (1 mm) is used so "
            "the screen is sensitive, with severity held at 'observation'. "
            "The 25 % relative path additionally requires a T wave of at "
            "least 2 mm: against a 1 mm T wave every U wave is "
            "'disproportionate', which describes the T wave rather than the "
            "U wave. See clinical_rules/u_wave.py."
        ),
    },
    {
        "topic": "inverted U wave detection",
        "baseline": "not quantified",
        "textbook_alternative": "negative U wave in a lead with an upright T wave",
        "resolution": "adopted_alternative_with_added_guards",
        "rationale": (
            "Highly specific for ischemia and therefore expensive to get "
            "wrong. Two guards were added beyond the textbook definition "
            "after both fired as false positives on LUDB: the deflection is "
            "gated on prominence rather than amplitude (amplitude is "
            "measured against the beat baseline, so post-T drift clears it "
            "with no wave underneath), and an isoelectric T-U segment of at "
            "least 20 ms is required (without it the terminal negative lobe "
            "of a biphasic T wave is indistinguishable from an inverted U). "
            "See u_wave.py and clinical_rules/u_wave.py."
        ),
    },
    {
        "topic": "pathological Q-wave depth",
        "baseline": ">25 % of the R wave in the lead",
        "textbook_alternative": ">2 mm absolute depth",
        "resolution": "baseline",
        "rationale": (
            "Pre-existing decision, recorded here for completeness: the "
            "absolute-depth form false-triggers wherever the QRS is large "
            "(LVH), so the ratio is used and absolute depth only serves as a "
            "fallback when R amplitude is unmeasurable."
        ),
    },
    {
        "topic": "hyperacute vs hyperkalemic tall T wave",
        "baseline": "not separated",
        "textbook_alternative": (
            "hyperacute T is broad and disproportionate to the QRS; "
            "hyperkalemic T is narrow, symmetric and peaked"
        ),
        "resolution": "adopted_alternative",
        "rationale": (
            "Amplitude cannot separate the two -- both are tall -- so the "
            "rule keys on T/QRS disproportion plus a >=160 ms base, and "
            "inside the 160-200 ms overlap band symmetry decides. The "
            "engine additionally suppresses the hyperacute statement when "
            "the hyperkalemia screen or an ST-elevation pattern has already "
            "matched. See clinical_rules/t_morphology.py."
        ),
    },
    {
        "topic": "multifocal atrial rhythm vs atrial fibrillation",
        "baseline": "not implemented",
        "textbook_alternative": (
            ">=3 P-wave morphologies with varying PP/PR/RR; >=100 bpm is MAT"
        ),
        "resolution": "adopted_alternative_with_added_guards",
        "rationale": (
            "AF is the differential that matters and it is not excluded by "
            "the textbook criteria alone: fibrillatory waves cluster into "
            "apparent morphologies as readily as real P waves. A record "
            "labelled AFIB in PTB-XL matched on the textbook criteria, so "
            "the rule additionally requires the organised-P ratio to reach "
            "0.60 -- the presence of discrete P waves is what actually "
            "separates MAT from AF. Morphology clusters must also hold two "
            "beats each, so measurement scatter cannot manufacture foci. "
            "See clinical_rules/atrial_rhythm.py."
        ),
    },
]

SOURCE_AHA_CONDUCTION_2009 = {
    "authority": "AHA/ACCF/HRS",
    "document": "ECG Standardization Part III",
    "section": "Intraventricular conduction disturbances",
    "version": "2009",
}
SOURCE_AHA_QT_2009 = {
    "authority": "AHA/ACCF/HRS",
    "document": "ECG Standardization Part IV",
    "section": "QT interval",
    "version": "2009",
}
SOURCE_UDMI_2018 = {
    "authority": "ESC/ACC/AHA/WHF",
    "document": "Fourth Universal Definition of Myocardial Infarction",
    "section": "ECG manifestations",
    "version": "2018",
}
SOURCE_AHA_HYPERTROPHY_2009 = {
    "authority": "AHA/ACCF/HRS",
    "document": "ECG Standardization Part V",
    "section": "Cardiac chamber hypertrophy",
    "version": "2009",
}
SOURCE_AHA_REPOLARIZATION_2009 = {
    "authority": "AHA/ACCF/HRS",
    "document": "ECG Standardization Part IV",
    "section": "ST segment, T and U waves, and QT interval",
    "version": "2009",
}
SOURCE_AHA_RHYTHM = {
    "authority": "AHA/ACC/HRS",
    "document": "ECG standardization and rhythm guidance",
    "section": "Ectopy, ventricular pre-excitation, and AV conduction",
    "version": "public-guideline projection",
}
SOURCE_AHA_LOW_VOLTAGE = {
    "authority": "AHA/ACCF/HRS",
    "document": "ECG Standardization Part II",
    "section": "Diagnostic terms for low QRS voltage",
    "version": "2009",
}
