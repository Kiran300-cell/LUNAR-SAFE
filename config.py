from pathlib import Path
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
V12_DIR = DATA_DIR / "v12"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
ASSET_DIR = APP_DIR / "assets"
SOURCE_URL = "https://astrogeology.usgs.gov/search/map/lunar_lro_nac_haworth_photoclinometry_dem_1m"
DEM_DOWNLOAD_URL = "https://planetarymaps.usgs.gov/mosaic/Lunar_Photoclinometry/Haworth-SfS/Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif"
HR_PATCH, SCALE = 100, 5
MODEL_LABELS = ["Bicubic", "Bilinear", "CNN_A", "CNN_B"]
