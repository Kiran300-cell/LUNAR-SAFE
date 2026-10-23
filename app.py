from io import BytesIO
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter
import streamlit as st
from config import ASSET_DIR, SOURCE_URL, V12_DIR
from data_loader import load_v12, read_output_archive
from metrics import METRICS, slope_degrees, hazard_scores, validate_metric_table
from terrain_analysis import inspect_geotiff
from visualizations import comparison_chart, heatmap, training_chart
from exports import workbook_bytes, pdf_bytes

st.set_page_config(page_title="LUNAR-SAFE | Terrain Intelligence",page_icon="◐",layout="wide",initial_sidebar_state="expanded")
st.markdown("<style>"+(Path(__file__).parent/"styles.css").read_text(encoding="utf-8")+"</style>",unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def load_dashboard_data():
    return load_v12()

d=load_dashboard_data(); metrics=d["metrics"]; summary=d["summary"]
if summary.empty and not metrics.empty: summary=metrics.groupby("model",as_index=False).mean(numeric_only=True)
main_models=["Bicubic","Bilinear","CNN_A","CNN_B"]
main_summary=summary[summary.model.isin(main_models)].copy() if "model" in summary else pd.DataFrame()
def navigate(target):
    st.session_state["nav_page"]=target

with st.sidebar:
    st.markdown('<div class="eyebrow">LUNAR-SAFE · V12</div>',unsafe_allow_html=True)
    st.markdown("### ◐ Terrain Intelligence")
    st.caption("HAWORTH · RESEARCH PROTOTYPE")
    page=st.radio("Navigation",["Mission Overview","Terrain Explorer","AI Workbench","Performance Lab","Hazard Analysis","Statistical Validation","Training Monitor","Dataset & QC","Export Center","Research Notes"],label_visibility="collapsed",key="nav_page")
    st.divider()
    st.markdown("**Session status**")
    st.markdown("🟢 Evaluation data available" if not metrics.empty else "🟠 Evaluation data missing")
    st.markdown("🟢 CNN checkpoints packaged")
    st.caption("GPU availability checked when inference starts.")
    st.divider()
    threshold=st.select_slider("Hazard proxy threshold",options=[5,10,15],value=10,format_func=lambda x:f"{x}°")
    st.caption("Threshold labels are exploratory proxies.")
    st.caption("V12 · synthetic 5× · single region")
st.markdown('<div class="eyebrow">MISSION CONTROL / HAWORTH REGION</div>',unsafe_allow_html=True)
st.title(page)
h1,h2,h3,h4,h5=st.columns([1,1,1,1.7,.9])
h1.caption("SESSION · V12 benchmark")
h2.caption("DATASET · Haworth SfS DEM")
h3.caption(f"MODEL · {len(d['checkpoints']) if not d['checkpoints'].empty else 0} checkpoints packaged")
validation="PASS" if d["run_manifest"].get("consistency_check")=="PASS for all patches and splits" and not d["audit"].empty else "NOT VERIFIED"
h4.caption(f"DATA VALIDATION · {validation}  ·  LAST V12 RUN · {d['run_manifest'].get('run_utc','not recorded')}")
h5.button("Export report",use_container_width=True,on_click=navigate,args=("Export Center",))

def kpi(label,value,note):
    st.markdown(f'<div class="kpi" title="{label}: {note}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
def model_value(model,key):
    rows=summary[summary.model==model] if "model" in summary else pd.DataFrame()
    return float(rows.iloc[0][key]) if not rows.empty and key in rows else float("nan")
def make_csv(frame): return frame.to_csv(index=False).encode("utf-8")
def render_metric(df,key):
    if key not in df: st.info(f"Metric not present: {key}"); return
    st.plotly_chart(comparison_chart(df,key),use_container_width=True,config={"displaylogo":False})
def show_limitations():
    st.markdown('<div class="callout"><b>Scientific scope</b><br>Results use synthetic 5× degradation from one Haworth DEM. Independent real 5 m validation is unavailable. Vertical units are unconfirmed; source artifacts may affect reconstruction. Slope labels are hazard proxies, not certified landing-safety labels. Higher detail does not establish mission safety.</div>',unsafe_allow_html=True)

if page=="Mission Overview":
    st.markdown('<div class="hero"><div class="eyebrow">LUNAR-SAFE / RESEARCH PLATFORM</div><h1>Lunar terrain,<br>made legible.</h1><p>Deep learning for terrain reconstruction and slope-based hazard assessment in the Haworth region. Existing V12 results are shown with their data provenance and limits.</p></div>',unsafe_allow_html=True)
    b1,b2,b3,b4=st.columns(4)
    with b1:
        st.button("Launch Terrain Analysis",use_container_width=True,on_click=navigate,args=("Terrain Explorer",))
    with b2:
        st.button("Compare Models",use_container_width=True,on_click=navigate,args=("Performance Lab",))
    with b3:
        st.button("View Research Results",use_container_width=True,on_click=navigate,args=("Statistical Validation",))
    with b4:
        st.button("Export Evaluation Report",use_container_width=True,on_click=navigate,args=("Export Center",))
    vals=[("Best relative elevation MAE",model_value("CNN_A","relative_elevation_MAE_m"),"CNN-A · lower is better"),
          ("Best relative elevation RMSE",model_value("CNN_A","relative_elevation_RMSE_m"),"CNN-A · lower is better"),
          ("Best slope MAE",min([model_value(x,"slope_MAE_deg") for x in main_models if np.isfinite(model_value(x,"slope_MAE_deg"))],default=np.nan),"CNN-A · degrees"),
          ("Highest F1 at 10°",max([model_value(x,"F1_at_10deg") for x in main_models if np.isfinite(model_value(x,"F1_at_10deg"))],default=np.nan),"CNN-A · higher is better"),
          ("Lowest false-safe rate",min([model_value(x,"false_safe_rate_at_10deg") for x in main_models if np.isfinite(model_value(x,"false_safe_rate_at_10deg"))],default=np.nan),"CNN-A · lower is better"),
          ("Held-out test patches",d["run_manifest"].get("test_patches_per_model","—"),"per method · recorded in run manifest")]
    cols=st.columns(6)
    for c,(label,val,note) in zip(cols,vals):
        with c:
            shown=f"{val:.5f}" if isinstance(val,(float,int,np.floating)) and np.isfinite(val) else str(val)
            kpi(label,shown,note)
    st.caption("Elevation values are relative, patch-centered errors. The historic V12 column names include “m”, but source vertical units are not confirmed; these are not presented as absolute metre errors.")
    st.markdown("#### Evaluation status")
    s1,s2,s3=st.columns(3)
    s1.success("V12 evaluation files loaded" if not metrics.empty else "Evaluation files missing")
    s2.success("Six V12 CNN checkpoints included")
    s3.success("Consistency audit read" if not d["audit"].empty else "Consistency audit missing")
    if not d["audit"].empty:
        st.dataframe(d["audit"],hide_index=True,use_container_width=True)
    show_limitations()

elif page=="Terrain Explorer":
    st.markdown("Inspect an uploaded single-band GeoTIFF. The display is downsampled for responsiveness; raster metadata is preserved for the inspection panel.")
    file=st.file_uploader("Upload lunar DEM GeoTIFF",type=["tif","tiff"],key="explore_tif")
    if file:
        try:
            info=inspect_geotiff(file)
            a=info.pop("array")
            col1,col2=st.columns([1.45,1])
            with col1:
                st.plotly_chart(heatmap(a,"Uploaded DEM · display sample"),use_container_width=True)
                row=st.slider("Cross-section row",0,a.shape[0]-1,value=a.shape[0]//2)
                fig=go.Figure(go.Scatter(x=np.arange(a.shape[1]),y=a[row],mode="lines",line={"color":"#42d9f5"}))
                fig.update_layout(title="Terrain profile",xaxis_title="Sampled column",yaxis_title="Elevation value",template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)
            with col2:
                st.markdown("#### Raster metadata")
                for label,key in [("Dimensions","shape"),("CRS","crs"),("Pixel size x","pixel_x"),("Pixel size y","pixel_y"),("NoData","nodata"),("Valid sample fraction","valid_fraction"),("Min / median / max","min")]:
                    st.write(f"**{label}:** {info.get(key)}")
                if not info["is_projected"]: st.warning("A projected CRS is required; physical slope interpretation is unavailable.")
            z=a[::max(1,a.shape[0]//120),::max(1,a.shape[1]//120)]
            exaggeration=st.slider("Vertical exaggeration (display only)",0.05,1.0,0.28,0.05)
            fig3=go.Figure(go.Surface(z=z,colorscale="Gray",colorbar={"title":"elevation"}))
            fig3.update_layout(title="Interactive 3D surface · display sample",scene={"aspectratio":{"x":1,"y":1,"z":exaggeration},"zaxis_title":"Elevation · display exaggeration"},template="plotly_dark")
            st.plotly_chart(fig3,use_container_width=True)
        except Exception as e: st.error(f"Raster validation failed: {e}")

elif page=="AI Workbench":
    st.markdown("### Synthetic benchmark models and uploaded-data inference")
    st.info("CNN-A and CNN-B use the V12 notebook architecture and packaged checkpoints. Uploaded DEM inference uses the notebook’s 5×5 averaging, 100×100 tiling, and LR-mean centering. Transfer to other areas is unvalidated.")
    model_file=st.file_uploader("Upload projected, single-band DEM for inference",type=["tif","tiff"],key="infer_tif")
    mode=st.selectbox("CNN objective",["A","B"],format_func=lambda x:f"CNN-{x}")
    seed=st.selectbox("Checkpoint seed",[42,143,244])
    if model_file and st.button("Run reconstruction",type="primary"):
        with st.spinner("Processing 100×100 tiles…"):
            try:
                from model_inference import infer_geotiff_bytes
                result,info=infer_geotiff_bytes(model_file.getvalue(),mode,seed)
                st.session_state["inference_result"]=result; st.session_state["inference_info"]=info
                st.success(f"Reconstruction completed on {info['device']} · {info['tiles_processed']} valid tiles")
            except Exception as e: st.error(str(e))
    if "inference_result" in st.session_state:
        info=st.session_state["inference_info"]
        st.write(info)
        st.download_button("Download reconstructed GeoTIFF",st.session_state["inference_result"],file_name=f"lunar_safe_cnn_{mode}_seed{seed}.tif",mime="image/tiff")
    if d["checkpoints"].empty: st.warning("Checkpoint metadata unavailable; packaged model files are still available.")
    else: st.dataframe(d["checkpoints"],hide_index=True,use_container_width=True)
    show_limitations()

elif page=="Performance Lab":
    st.markdown("#### Held-out test comparison · existing V12 evaluation")
    if main_summary.empty: st.warning("V12 summary metrics are unavailable.")
    else:
        tabs=st.tabs(["Elevation MAE","Elevation RMSE","Slope MAE","Hazard F1","False-safe"])
        keys=list(METRICS)
        for tab,key in zip(tabs,keys):
            with tab: render_metric(main_summary,key)
        st.dataframe(main_summary[["model"]+list(METRICS)].rename(columns={"model":"Method"}),hide_index=True,use_container_width=True)
        st.download_button("Download shown metrics CSV",make_csv(main_summary),file_name="lunar_safe_v12_metrics.csv",mime="text/csv")
    st.markdown("#### Existing V12 figures")
    names=["elevation_mae_comparison.png","f1_10deg_comparison.png","slope_mae_comparison.png","heldout_reconstruction_comparison.png"]
    cols=st.columns(2)
    for i,name in enumerate(names):
        p=ASSET_DIR/name
        if p.exists():
            with cols[i%2]: st.image(str(p),caption=name.replace("_"," ").replace(".png",""))
    st.markdown("#### Held-out patch inspection")
    st.caption("Upload the V12 test_predictions_and_targets.npz file from the original outputs archive. Arrays are read by key; the dashboard does not copy the large array into the project.")
    npz_upload=st.file_uploader("V12 prediction and target arrays (.npz)",type=["npz"],key="prediction_npz")
    if npz_upload:
        try:
            token=(npz_upload.name,npz_upload.size,getattr(npz_upload,"file_id",None))
            if st.session_state.get("prediction_npz_token")!=token:
                prior=st.session_state.get("prediction_npz_object")
                if prior is not None:
                    try: prior.close()
                    except Exception: pass
                npz_upload.seek(0)
                st.session_state["prediction_npz_object"]=np.load(npz_upload,allow_pickle=False)
                st.session_state["prediction_npz_token"]=token
                st.session_state["prediction_arrays"]={}
            packed=st.session_state["prediction_npz_object"]
            arrays=st.session_state["prediction_arrays"]
            required={"Xtest","Ytest","pred_Bicubic","pred_Bilinear","pred_CNN_A","pred_CNN_B"}
            missing=required-set(packed.files)
            if missing: st.error("Required arrays are missing: "+", ".join(sorted(missing)))
            else:
                if packed["Xtest"].shape!=(600,1,20,20) or packed["Ytest"].shape!=(600,1,100,100):
                    st.error("Unexpected V12 array shapes; patch visualization disabled.")
                else:
                    for key in ["Xtest","Ytest","patch_rows","patch_cols"]:
                        if key in packed.files and key not in arrays: arrays[key]=packed[key]
                    patch_idx=st.slider("Test patch index",0,599,0)
                    pred_name=st.selectbox("Prediction",["Bicubic","Bilinear","CNN_A","CNN_B"],format_func=lambda x:x.replace("_","-"))
                    pred_key="pred_"+pred_name
                    if pred_key not in arrays: arrays[pred_key]=packed[pred_key]
                    lr=arrays["Xtest"][patch_idx,0]
                    ref=arrays["Ytest"][patch_idx,0]
                    pred=arrays[pred_key][patch_idx,0]
                    from plotly.subplots import make_subplots
                    fig=make_subplots(rows=2,cols=3,subplot_titles=["5 m input · centered","1 m reference · centered",pred_name+" · centered","Absolute relative-elevation error","Reference slope*","Prediction slope*"])
                    for j,(arr,title) in enumerate([(lr,""),(ref,""),(pred,""),(np.abs(pred-ref),""),(slope_degrees(ref,1,1),""),(slope_degrees(pred,1,1),"")]):
                        fig.add_trace(go.Heatmap(z=arr,colorscale="Turbo",showscale=False),row=j//3+1,col=j%3+1)
                    fig.update_layout(height=650,title=f"V12 test patch {patch_idx} · centered elevation views",template="plotly_dark")
                    st.plotly_chart(fig,use_container_width=True)
                    if "patch_rows" in arrays and "patch_cols" in arrays:
                        st.caption(f"Source raster patch origin: row {int(arrays['patch_rows'][patch_idx])}, column {int(arrays['patch_cols'][patch_idx])}.")
                    st.caption("*Slope views assume vertical units match the DEM horizontal metre grid; source vertical units are unconfirmed.")
        except Exception as e: st.error(f"Prediction array validation failed: {e}")
    show_limitations()

elif page=="Hazard Analysis":
    st.markdown("Adjust the limits below to screen the uploaded DEM. The map and area summary update as you change the controls. These classes are exploratory hazard proxies, not mission-safety classifications.")
    file=st.file_uploader("Upload projected GeoTIFF",type=["tif","tiff"],key="hazard_tif")
    st.warning("The V12 raster tags do not confirm vertical units. Calculated slope angles are conditional on matching vertical and horizontal units.")
    if file:
        assume=st.checkbox("Assume DEM elevation units match projected horizontal units (unverified)")
        try:
            info=inspect_geotiff(file); a=info.pop("array")
            if not info["is_projected"]: st.error("A projected CRS is required for slope analysis.")
            elif not assume: st.info("Confirm the unverified units assumption above to calculate a conditional slope proxy.")
            else:
                c1,c2=st.columns(2)
                with c1: slope_limit=st.slider("Maximum slope for a pass (degrees)",1.0,30.0,float(threshold),0.5,key="screen_slope")
                with c2: roughness_limit=st.slider("Maximum local roughness (elevation units)",0.0,10.0,1.0,0.1,key="screen_roughness")
                window=st.select_slider("Roughness neighborhood",options=[3,5,7,9],value=5,format_func=lambda n:f"{n} × {n} cells",key="screen_window")
                if not np.isfinite(a).all(): raise ValueError("This screening view requires a complete valid raster window without NoData cells.")
                s=slope_degrees(a,info["pixel_x"],info["pixel_y"])
                a32=a.astype(np.float32)
                local_mean=uniform_filter(a32,size=window,mode="nearest")
                local_sq_mean=uniform_filter(np.square(a32),size=window,mode="nearest")
                roughness=np.sqrt(np.maximum(local_sq_mean-np.square(local_mean),0))
                high_slope=s>slope_limit; high_rough=roughness>roughness_limit
                classes=np.zeros(a.shape,dtype=np.uint8)
                classes[high_slope|high_rough]=1
                classes[high_slope&high_rough]=2
                labels=classes>0
                k1,k2,k3,k4=st.columns(4)
                k1.metric("Screened area",f"{labels.mean()*100:.2f}%")
                k2.metric("Pass proxy",f"{(~labels).mean()*100:.2f}%")
                k3.metric("Slope flags",f"{high_slope.mean()*100:.2f}%")
                k4.metric("Roughness flags",f"{high_rough.mean()*100:.2f}%")
                st.caption(f"Evaluated on downsampled display grid ({a.shape[1]} × {a.shape[0]} cells); roughness is local elevation standard deviation over {window} × {window} cells. Approximate ground spacing: {info['pixel_x']:.2f} × {info['pixel_y']:.2f} map units per display cell.")
                fig=go.Figure(go.Heatmap(z=classes,colorscale=[[0,"#173c35"],[0.5,"#e9a23b"],[1,"#ed5e5e"]],zmin=0,zmax=2,colorbar={"title":"screen class","tickvals":[0,1,2],"ticktext":["Pass proxy","Review","High priority"]}))
                fig.update_layout(title="Live terrain screening · pass / review / high priority",template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)
                meta={"slope_threshold_degrees":slope_limit,"local_roughness_threshold_elevation_units":roughness_limit,"roughness_window_cells":window,"grid_shape":list(a.shape),"screened_cells":int(labels.sum()),"review_cells":int((classes==1).sum()),"high_priority_cells":int((classes==2).sum()),"screened_area_percent":float(labels.mean()*100),"source":file.name,"vertical_unit_assumption":"user-selected, unverified","classification":"Exploratory screening proxy; not a mission safety certification"}
                st.download_button("Download analysis metadata JSON",json.dumps(meta,indent=2),file_name="hazard_proxy_metadata.json",mime="application/json")
        except Exception as e: st.error(f"Slope analysis failed: {e}")
    if not d["pooled"].empty: st.dataframe(d["pooled"],hide_index=True,use_container_width=True)
    show_limitations()
elif page=="Statistical Validation":
    st.markdown("#### Paired spatial block-bootstrap vs bilinear baseline")
    boot=d["bootstrap"]
    if boot.empty: st.info("Bootstrap interval data were not present.")
    else:
        cols=["candidate","metric","difference_candidate_minus_baseline","ci_2_5","ci_97_5","n_paired_patches","n_spatial_blocks","direction","ci_excludes_zero"]
        st.dataframe(boot[cols],hide_index=True,use_container_width=True)
        key=st.selectbox("Contrast",boot.candidate.unique())
        view=boot[boot.candidate==key]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=view.difference_candidate_minus_baseline,y=view.metric,mode="markers",error_x={"type":"data","symmetric":False,"array":view.ci_97_5-view.difference_candidate_minus_baseline,"arrayminus":view.difference_candidate_minus_baseline-view.ci_2_5},marker={"color":"#42d9f5","size":10}))
        fig.add_vline(x=0,line_dash="dash",line_color="#f2b866")
        fig.update_layout(title=f"{key} − bilinear · 95% spatial block-bootstrap intervals",xaxis_title="Difference in patch means",template="plotly_dark")
        st.plotly_chart(fig,use_container_width=True)
    st.caption("Intervals resample 36 spatial blocks (2,000 replicates; seed 20261023). The intervals depend on block scale and do not establish generalization across lunar regions.")
    show_limitations()

elif page=="Training Monitor":
    hist=d["training"]
    if hist.empty: st.info("Training history is missing.")
    else:
        mode=st.selectbox("Model objective",["A","B"],format_func=lambda x:f"CNN-{x}")
        seed=st.selectbox("Training seed",[42,143,244],key="training_seed")
        st.plotly_chart(training_chart(hist,mode,seed),use_container_width=True)
        sel=hist[(hist["mode"]==mode)&(hist.seed==seed)]
        best=d["checkpoints"][(d["checkpoints"]["mode"]==mode)&(d["checkpoints"].seed==seed)] if not d["checkpoints"].empty else pd.DataFrame()
        a,b,c=st.columns(3)
        a.metric("Recorded epochs",len(sel))
        b.metric("Best checkpoint epoch",int(best.iloc[0].best_epoch) if not best.empty else "—")
        c.metric("Early stop",str(len(sel)<30))
        st.dataframe(sel,hide_index=True,use_container_width=True)
        st.caption("Checkpoint selection and early stopping use validation elevation L1 for both objectives. No hardware-performance claims are inferred.")
    show_limitations()

elif page=="Dataset & QC":
    manifest=d["dem_manifest"]; run=d["run_manifest"]
    st.markdown("#### Source DEM manifest")
    if manifest: st.json(manifest)
    else: st.warning("DEM manifest missing.")
    st.markdown("#### Integrity checks")
    checks=validate_metric_table(metrics) if not metrics.empty else {}
    if checks: st.json(checks)
    if not d["audit"].empty: st.dataframe(d["audit"],hide_index=True,use_container_width=True)
    st.write("Consistency check in run manifest:",run.get("consistency_check","unknown"))
    if d["missing"]: st.warning("Missing V12 artifacts: "+", ".join(d["missing"]))
    if not (V12_DIR/"test_predictions_and_targets.npz").exists():
        st.info("The large prediction/target NPZ was omitted from this source ZIP. Patch-level input/reference/prediction/error maps are disabled until that array is supplied; metric tables and saved figures remain available.")
    up=st.file_uploader("Inspect a user-supplied V12 outputs ZIP",type=["zip"])
    if up:
        items,errors=read_output_archive(up)
        for e in errors: st.error(e)
        st.success(f"Found {len(items)} supported output files.")
        st.write(", ".join(items.keys()) if items else "No recognized files.")
        for name,obj in items.items():
            if isinstance(obj,pd.DataFrame): st.write(name); st.dataframe(obj.head(10),hide_index=True)
    st.markdown(f"**Official source information:** [{SOURCE_URL}]({SOURCE_URL})")
    show_limitations()

elif page=="Export Center":
    st.markdown("Generate research exports from the files loaded in this app. Exports contain existing measurements only.")
    if main_summary.empty: st.warning("Metrics unavailable; reports cannot be built.")
    else:
        manifest={"dem_manifest":d["dem_manifest"],"run_manifest":d["run_manifest"],"config":d["config"]}
        patch_data=d["metrics"]
        patch_cols=[c for c in ["model","patch_index","row","col",*METRICS] if c in patch_data.columns]
        patch_table=patch_data[patch_data.model.isin(main_models)][patch_cols] if not patch_data.empty else pd.DataFrame()
        xlsx=workbook_bytes(summary,d["bootstrap"],d["training"],manifest,d["pooled"],patch_table)
        pdf=pdf_bytes(summary,d["bootstrap"],manifest,d["pooled"],d["training"],d["config"])
        c1,c2=st.columns(2)
        c1.download_button("Download Excel dashboard",xlsx,file_name="LUNAR_SAFE_V12_Evaluation.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
        c2.download_button("Download PDF research report",pdf,file_name="LUNAR_SAFE_V12_Report.pdf",mime="application/pdf",use_container_width=True)
        st.download_button("Download metrics CSV",make_csv(main_summary),file_name="LUNAR_SAFE_V12_metrics.csv",mime="text/csv")
        manifest_json=json.dumps(manifest,indent=2)
        st.download_button("Download provenance JSON",manifest_json,file_name="LUNAR_SAFE_V12_provenance.json",mime="application/json")
        bundle=BytesIO()
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("metrics.csv",make_csv(main_summary)); z.writestr("report.pdf",pdf); z.writestr("dashboard.xlsx",xlsx); z.writestr("provenance.json",manifest_json)
            if not d["bootstrap"].empty: z.writestr("bootstrap.csv",make_csv(d["bootstrap"]))
        st.download_button("Download report package ZIP",bundle.getvalue(),file_name="LUNAR_SAFE_V12_exports.zip",mime="application/zip")
    show_limitations()

elif page=="Research Notes":
    st.markdown("### Research motivation")
    st.write("Lunar landing-site assessment benefits from terrain descriptions at scales finer than many orbital products provide. This prototype evaluates learned reconstruction on a high-resolution Haworth photoclinometry DEM under a controlled synthetic degradation.")
    st.markdown("### Dataset and methods")
    st.write("USGS Haworth SfS DEM, nominal 1 m horizontal sampling. Non-overlapping 5×5 block averages form nominal 5 m inputs. Training, validation, and test patches occupy separated spatial column bands (600 each). Both CNNs use validation elevation L1 for checkpoint selection.")
    st.write("CNN-A optimizes elevation L1. CNN-B adds a first-order gradient L1 penalty with weight 0.10. Baselines are bilinear and bicubic interpolation. The reported error metrics are computed after patch centering.")
    st.markdown("### Results")
    if not main_summary.empty: st.dataframe(main_summary[["model"]+list(METRICS)],hide_index=True,use_container_width=True)
    st.markdown("### Limitations and reproducibility")
    st.write("Single Haworth region; synthetic degradation; no independent real 5 m validation; source artifacts may affect metrics; vertical units are unconfirmed; bootstrap uncertainty depends on spatial-block scale. Hazard labels are proxies only. No mission certification or real-world landing-safety claim is made.")
    st.write(f"Source reference: [{SOURCE_URL}]({SOURCE_URL})")
    st.write("Recorded V12 configuration, audit report, split metadata, paired bootstrap table, and checkpoint epoch records are included under data/v12.")
    show_limitations()
