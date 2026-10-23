# LUNAR-SAFE — Lunar Terrain Reconstruction & Hazard Analysis

A Streamlit research dashboard for the existing LUNAR-SAFE V12 evaluation and optional CNN inference on uploaded single-band GeoTIFFs. This is a research prototype, not a mission-certified system.

## Run locally

Use Python 3.13 for Windows. Python 3.14 may not have a compatible Rasterio wheel and can fail while trying to build GDAL.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

PyTorch CPU works for inference; for GPU acceleration, install a build suited to your CUDA platform using the official selector at https://pytorch.org/get-started/locally/. The V12 results dashboard does not need a GPU.

## Google Colab

Upload/extract the project ZIP, then run:

    %cd /content/LUNAR_SAFE
    %pip install -r requirements.txt
    !streamlit run app.py --server.port 8501

Use Colab port forwarding or a trusted Streamlit tunnel if remote access is needed. No tunnel is configured by this project.

## Included V12 data

data/v12 contains lightweight real V12 metrics, training history, manifests, consistency audit, patch metadata, checkpoint selection records, spatial block bootstrap intervals and pooled hazard counts. data/checkpoints contains six actual seed checkpoints extracted from the supplied V12 output archive. The large test_predictions_and_targets.npz (about 237 MB) and original DEM are external; the app does not silently substitute demo data. Saved V12 figures are under assets.

Upload the original output ZIP on Dataset & QC to inspect supported CSV/JSON files. Patch-level input/reference/prediction/error views are available in Performance Lab by uploading the original test_predictions_and_targets.npz (about 237 MB). The NPZ is shape-checked and read by requested array; it is not bundled in this lightweight ZIP.

## Science and preprocessing

- V12 compares bicubic, bilinear, CNN-A and CNN-B on 600 spatially separated test patches per method.
- Synthetic low-resolution inputs use non-overlapping 5×5 averaging. Each 100×100 reference tile becomes a 20×20 input.
- CNN-A uses elevation L1; CNN-B adds a first-order gradient L1 penalty (weight 0.10). Both select checkpoints with validation elevation L1.
- Test elevation errors are computed after patch-centering. They are relative elevation errors; vertical units in the source tags are unconfirmed.
- Upload inference applies the notebook's residual CNN architecture, 5×5 block averaging and tile-mean centering. Transfer outside Haworth is unvalidated. Partial tiles and tiles containing NoData are skipped.

## Data lineage

The source CRS is a lunar polar stereographic projection. Source manifest shape: 12,060 × 11,660; nominal horizontal pixel spacing: 1 m. Band vertical units are not confirmed. Official source information: https://astrogeology.usgs.gov/search/map/lunar_lro_nac_haworth_photoclinometry_dem_1m

## Limitations

Synthetic degradation only; one region; no independent real 5 m validation; source DEM artifacts may affect reconstruction and evaluation; vertical units remain unconfirmed; threshold labels are hazard proxies, not certified landing safety labels. Better spatial detail does not establish improved real-world landing safety. Spatial block-bootstrap intervals use 36 blocks and depend on the block scale.

## Project map

app.py, config.py, data_loader.py, metrics.py, terrain_analysis.py, model_inference.py, visualizations.py, exports.py, requirements.txt, tests/, assets/, data/v12/ and data/checkpoints/.

## SEO and deployment

`docs/index.html` is the search-friendly public project landing page. It explains the dashboard and its scientific limits; its repository and hosted-app links must be updated after deployment. Follow [DEPLOYMENT.md](DEPLOYMENT.md) to publish the landing page with GitHub Pages and run the interactive Streamlit dashboard on Streamlit Community Cloud. Public indexing and Google rankings are not guaranteed.
