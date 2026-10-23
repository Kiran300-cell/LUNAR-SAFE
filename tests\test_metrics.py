import numpy as np
from metrics import hazard_scores, slope_degrees
def test_flat_surface_has_zero_slope():
    assert np.allclose(slope_degrees(np.ones((5,5)),1,1),0)
def test_hazard_proxy_counts_and_rates():
    s=hazard_scores(np.array([[0,12],[15,0]]),np.array([[0,0],[15,18]]),10)
    assert (s["TP"],s["FP"],s["FN"],s["TN"])==(1,1,1,1)
    assert s["F1"]==0.5 and s["false_safe_rate"]==0.5
