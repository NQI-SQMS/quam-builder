from typing import List, Optional

from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qubit.fixed_frequency_transmon import (
    FixedFrequencyTransmon,
)

__all__ = ["SrfTransmon"]


@quam_dataclass
class SrfTransmon(FixedFrequencyTransmon):
    """FixedFrequencyTransmon for SRF qubit+cavity experiments.

    EF-transition pulses (EF_x180, EF_x90, etc.) are stored on the same xy
    channel as the GE pulses.  The QUA program switches to the EF frequency by
    calling qubit.xy.update_frequency(qubit.xy.intermediate_frequency + qubit.anharmonicity)
    before playing an EF gate, then resets afterwards.

    GEF readout classifier parameters (in volts, set by node 15_iq_blobs_gef):
        gef_rotation_angle      : 1D-threshold rotation angle (rad), aligns g→e axis with I axis.
        gef_threshold_low       : Lower 1D threshold in volts.
        gef_threshold_high      : Upper 1D threshold in volts.
        gef_threshold_sorted_order : State indices [0=g,1=e,2=f] sorted along the rotated I axis.
        gef_lda_sigma           : 2×2 pooled LDA covariance matrix (V²).
        gef_lda_priors          : LDA class priors [p_g, p_e, p_f].
        gef_confusion_matrix_lda: 3×3 LDA confusion matrix (rows=prepared, cols=measured).
        gef_d_ge_V              : g-e centroid distance in volts.
        gef_d_gf_V              : g-f centroid distance in volts.
        gef_d_ef_V              : e-f centroid distance in volts.
        gef_sigma_rms_V         : RMS blob width from LDA pooled covariance (volts).
    """

    # EF dispersive shift (set by node 20b_dispersive_shift_gef)
    chi_ef: Optional[float] = None
    """EF dispersive shift chi_ef = f_resonator(|f>) - f_resonator(|e>) [Hz]."""

    # 1D rotated-threshold classifier
    gef_rotation_angle: Optional[float] = None
    gef_threshold_low: Optional[float] = None
    gef_threshold_high: Optional[float] = None
    gef_threshold_sorted_order: Optional[List[int]] = None

    # LDA classifier
    gef_lda_sigma: Optional[List[List[float]]] = None
    gef_lda_priors: Optional[List[float]] = None
    gef_confusion_matrix_lda: Optional[List[List[float]]] = None

    # Blob separation metrics
    gef_d_ge_V: Optional[float] = None
    gef_d_gf_V: Optional[float] = None
    gef_d_ef_V: Optional[float] = None
    gef_sigma_rms_V: Optional[float] = None
