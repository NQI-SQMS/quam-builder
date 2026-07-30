import numpy as np
import xarray as xr

__all__ = ["apply_confusion_matrix_correction"]


def apply_confusion_matrix_correction(ds: xr.Dataset, qubits) -> xr.Dataset:
    """Correct averaged state probabilities for ge readout errors using stored confusion matrices.

    Args:
        ds: Dataset containing a 'state' data variable with a 'qubit' dimension.
        qubits: Iterable of qubit objects (e.g. node.namespace["qubits"]).

    Returns:
        Dataset with corrected 'state' values, or the original dataset if no
        confusion matrices are found or 'state' is not present.
    """
    if "state" not in ds.data_vars:
        return ds

    cm_inv_map = {}
    for qubit in qubits:
        cm = getattr(qubit.resonator, "confusion_matrix", None)
        if cm is not None:
            cm_inv_map[qubit.name] = np.linalg.inv(np.array(cm))

    if not cm_inv_map:
        return ds

    corrected = ds.state.copy(deep=True)
    for q_name, cm_inv in cm_inv_map.items():
        s = ds.state.sel(qubit=q_name).values.astype(float)
        p_meas = np.stack([1.0 - s, s], axis=-1)
        p_true = (cm_inv @ p_meas.T).T
        corrected.loc[dict(qubit=q_name)] = p_true[..., 1]

    return ds.assign(state=corrected)
