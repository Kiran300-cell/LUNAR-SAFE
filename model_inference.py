from pathlib import Path
import numpy as np
from config import CHECKPOINT_DIR
def _model_class():
    try:
        import torch, torch.nn as nn, torch.nn.functional as F
    except ImportError as e: raise RuntimeError("Install PyTorch to use CNN inference. Existing results remain available.") from e
    class TerrainResidualCNN(nn.Module):
        def __init__(self,scale=5,width=64,depth=8):
            super().__init__(); self.scale=scale
            layers=[nn.Conv2d(1,width,3,padding=1),nn.ReLU(inplace=True)]
            for _ in range(depth-2): layers += [nn.Conv2d(width,width,3,padding=1),nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(width,scale*scale,3,padding=1),nn.PixelShuffle(scale)]
            self.body=nn.Sequential(*layers)
        def forward(self,x):
            return F.interpolate(x,scale_factor=self.scale,mode="bilinear",align_corners=False)+self.body(x)
    return torch,TerrainResidualCNN
def load_model(mode="A",seed=42):
    torch,Model=_model_class(); device="cuda" if torch.cuda.is_available() else "cpu"
    path=CHECKPOINT_DIR/f"cnn_{mode}_seed{seed}.pt"
    if not path.exists(): raise FileNotFoundError(path.name)
    model=Model().to(device); model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True); model.eval()
    return model,device
def reconstruct_uploaded_dem(dem,mode="A",seed=42,patch=100,scale=5):
    torch,_=_model_class(); model,device=load_model(mode,seed)
    src=np.asarray(dem,dtype=np.float32)
    if src.ndim!=2: raise ValueError("Expected a single-band DEM.")
    out=np.full(src.shape,np.nan,dtype=np.float32); count=0
    for r in range(0,src.shape[0]-patch+1,patch):
        for c in range(0,src.shape[1]-patch+1,patch):
            tile=src[r:r+patch,c:c+patch]
            if not np.isfinite(tile).all(): continue
            lr=tile.reshape(patch//scale,scale,patch//scale,scale).mean(axis=(1,3),dtype=np.float64).astype(np.float32)
            offset=float(lr.mean()); x=torch.as_tensor((lr-offset)[None,None],device=device)
            with torch.inference_mode(): pred=(model(x)[0,0].cpu().numpy()+offset).astype(np.float32)
            out[r:r+patch,c:c+patch]=pred; count+=1
    if not count: raise ValueError("No complete, finite 100×100 tiles available.")
    return out,{"tiles_processed":count,"device":device,"model":f"CNN-{mode}","seed":seed,"scale_factor":scale,"patch_size":patch}

def infer_geotiff_bytes(upload_bytes,mode="A",seed=42):
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.windows import Window
    except ImportError as e: raise RuntimeError("Install rasterio to export GeoTIFF results.") from e
    torch,_=_model_class(); model,device=load_model(mode,seed)
    with MemoryFile(upload_bytes) as src_mem:
        with src_mem.open() as src:
            if src.count!=1: raise ValueError("Use a single-band DEM GeoTIFF.")
            if src.crs is None or not src.crs.is_projected: raise ValueError("Inference requires a projected CRS.")
            if src.width*src.height>50000000: raise ValueError("Raster exceeds the 50-megapixel safety limit; upload a cropped area.")
            profile=src.profile.copy(); profile.update(dtype="float32",count=1,compress="deflate",nodata=np.nan)
            with MemoryFile() as out_mem:
                with out_mem.open(**profile) as dst:
                    count=0
                    for r in range(0,src.height-99,100):
                        for c in range(0,src.width-99,100):
                            tile=src.read(1,window=Window(c,r,100,100),masked=True).filled(np.nan).astype(np.float32)
                            if not np.isfinite(tile).all(): continue
                            lr=tile.reshape(20,5,20,5).mean(axis=(1,3),dtype=np.float64).astype(np.float32)
                            offset=float(lr.mean()); x=torch.as_tensor((lr-offset)[None,None],device=device)
                            with torch.inference_mode(): pred=(model(x)[0,0].cpu().numpy()+offset).astype(np.float32)
                            dst.write(pred,1,window=Window(c,r,100,100)); count+=1
                    if count==0: raise ValueError("No complete, finite 100×100 tiles available.")
                return out_mem.read(),{"tiles_processed":count,"device":device,"model":f"CNN-{mode}","seed":seed}
