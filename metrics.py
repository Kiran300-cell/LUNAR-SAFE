import numpy as np
METRICS={"relative_elevation_MAE_m":("Relative elevation MAE","lower"),
"relative_elevation_RMSE_m":("Relative elevation RMSE","lower"),"slope_MAE_deg":("Slope MAE","lower"),
"F1_at_10deg":("F1 at 10°","higher"),"false_safe_rate_at_10deg":("False-safe rate","lower")}
def slope_degrees(dem,pixel_x,pixel_y,vertical_scale=1.0):
    gy,gx=np.gradient(np.asarray(dem,dtype=np.float64)*vertical_scale,float(pixel_y),float(pixel_x))
    return np.degrees(np.arctan(np.hypot(gx,gy)))
def hazard_scores(reference_slope,predicted_slope,threshold):
    ref=np.asarray(reference_slope)>=threshold; pred=np.asarray(predicted_slope)>=threshold
    tp=int((ref&pred).sum()); fp=int((~ref&pred).sum()); fn=int((ref&~pred).sum()); tn=int((~ref&~pred).sum())
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"F1":2*p*r/(p+r) if p+r else 0.,"false_safe_rate":fn/(tp+fn) if tp+fn else float("nan")}
def validate_metric_table(frame):
    req={"model","patch_index",*METRICS}; cols=set(frame.columns)
    return {"required_columns":req.issubset(cols),"rows":len(frame),"models":int(frame.model.nunique()) if "model" in cols else 0,
      "duplicate_rows":bool(frame.duplicated(["model","patch_index"]).any()) if {"model","patch_index"}.issubset(cols) else None}
