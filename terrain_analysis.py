import numpy as np
def inspect_geotiff(upload,max_side=800):
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as e: raise RuntimeError("Install rasterio to inspect GeoTIFF files.") from e
    with MemoryFile(upload.getvalue()) as mem, mem.open() as src:
        if src.count<1: raise ValueError("Raster has no bands.")
        ratio=max(src.width/max_side,src.height/max_side,1)
        h,w=max(1,int(src.height/ratio)),max(1,int(src.width/ratio))
        a=src.read(1,out_shape=(h,w),masked=True,resampling=rasterio.enums.Resampling.average).filled(np.nan).astype(np.float32)
        if not np.isfinite(a).any(): raise ValueError("No finite raster cells found.")
        return {"array":a,"shape":(src.height,src.width),"crs":str(src.crs) if src.crs else None,
          "transform":src.transform,"pixel_x":abs(float(src.transform.a))*src.width/w,"pixel_y":abs(float(src.transform.e))*src.height/h,
          "is_projected":bool(src.crs and src.crs.is_projected),
          "nodata":src.nodata,"dtype":src.dtypes[0],"bounds":tuple(src.bounds),"valid_fraction":float(np.isfinite(a).mean()),
          "min":float(np.nanmin(a)),"median":float(np.nanmedian(a)),"max":float(np.nanmax(a))}
def slope_from_array(a,dx,dy):
    from metrics import slope_degrees
    if not np.isfinite(a).all(): raise ValueError("Slope calculation needs finite pixels. Upload a smaller, complete valid area.")
    return slope_degrees(a,dx,dy)
