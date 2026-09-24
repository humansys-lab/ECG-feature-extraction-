from __future__ import annotations

from typing import Dict


SNOMED_BY_STATEMENT: Dict[str, str] = {
    "sinus_rhythm": "426783006",
    "sinus_bradycardia": "426177001",
    "sinus_tachycardia": "427084000",
    "atrial_fibrillation": "164889003",
    "atrial_fibrillation_pattern": "164889003",
    "probable_atrial_fibrillation_pattern": "164889003",
    "atrial_flutter_pattern": "164890007",
    "atrial_tachycardia": "713422000",
    "premature_atrial_complexes": "284470004",
    "premature_ventricular_complexes": "17338001",
    "first_degree_av_block": "270492004",
    "first_degree_av_delay": "270492004",
    "possible_first_degree_av_delay": "270492004",
    "second_degree_av_block": "195042002",
    "second_degree_av_block_pattern": "195042002",
    "complete_av_block": "27885002",
    "left_bundle_branch_block": "164909002",
    "lbbb_pattern": "164909002",
    "right_bundle_branch_block": "59118001",
    "rbbb_pattern": "59118001",
    "bifascicular_block_pattern": "6374002",
    "ventricular_preexcitation_pattern": "195060002",
    "left_ventricular_hypertrophy": "164873001",
    "lvh_voltage_criteria": "164873001",
    "right_ventricular_hypertrophy": "89792004",
    "rvh_pattern": "89792004",
    "low_voltage_limb_leads": "251147008",
    "low_voltage_precordial_leads": "251148003",
    "prolonged_qt": "111975006",
    "markedly_prolonged_qt": "111975006",
    "st_depression": "429622005",
    "st_elevation": "164930006",
    "t_wave_abnormality": "164934002",
    "pathological_q_waves": "164917005",
    "prior_infarct_q_wave_pattern": "164917005",
}

# Statement codes deliberately left unmapped, because no SNOMED CT concept was
# verified for them. `codes_for` returns an empty mapping for these, which is
# the correct behaviour -- emitting a guessed identifier would be worse than
# emitting none. Fill in only against an actual terminology browser.
UNMAPPED_STATEMENT_CODES = (
    "trifascicular_block_pattern",
    "lafb_pattern",
    "lpfb_pattern",
    "poor_r_wave_progression",
    "reversed_r_wave_progression",
    "precordial_rotation",
    "prominent_u_wave",
    "inverted_u_wave",
    "hyperacute_t_wave_pattern",
    "giant_negative_t_wave",
    "junctional_rhythm_pattern",
    "ectopic_atrial_rhythm_pattern",
    "multifocal_atrial_rhythm",
    "multifocal_atrial_tachycardia",
)


def codes_for(statement_code: str | None) -> Dict[str, str]:
    if not statement_code:
        return {}
    snomed = SNOMED_BY_STATEMENT.get(str(statement_code))
    return {"SNOMED_CT": snomed} if snomed else {}
