from pathlib import Path
import json, zipfile
import pandas as pd
from config import V12_DIR

FILES = {"metrics":"test_metrics_per_patch.csv","summary":"test_metrics_summary.csv","training":"training_history.csv",
         "bootstrap":"paired_spatial_block_bootstrap_vs_bilinear.csv","audit":"lr_hr_consistency_audit.csv",
         "patches":"test_patch_metadata_used.csv","blocks":"spatial_block_sizes.csv","checkpoints":"best_checkpoints.csv",
         "pooled":"pooled_hazard_confusion_metrics.csv"}
def load_v12():
    d={}
    for key,name in FILES.items():
        p=V12_DIR/name
        d[key]=pd.read_csv(p) if p.exists() else pd.DataFrame()
    for key,name in [("dem_manifest","dem_manifest.json"),("run_manifest","run_manifest.json"),("config","config.json")]:
        p=V12_DIR/name
        try: d[key]=json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError,json.JSONDecodeError): d[key]={}
    d["missing"]=[n for k,n in FILES.items() if d[k].empty]
    return d
def read_output_archive(upload):
    found,errors={},[]
    allow=set(FILES.values())|{"dem_manifest.json","run_manifest.json","config.json"}
    try:
        with zipfile.ZipFile(upload) as z:
            for member in z.namelist():
                name=Path(member).name
                if name not in allow: continue
                with z.open(member) as f:
                    found[name]=pd.read_csv(f) if name.endswith(".csv") else json.loads(f.read().decode("utf-8"))
    except (zipfile.BadZipFile,OSError,ValueError) as e: errors.append(str(e))
    return found,errors
