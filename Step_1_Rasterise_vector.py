
import numpy as np
import geopandas as gpd
import rasterio.features
import rioxarray as rxr
import xarray

from itertools import repeat
from joblib import Parallel, delayed
from tqdm.auto import tqdm



# Read the reference raster data
ref = rxr.open_rasterio("data/NLUM_2010-11_mask.tif", chunks=True)

ref_transform = ref.rio.transform()
ref_crs = ref.rio.crs
ref_shape = ref.rio.shape

# Read the SNES biodiversity data
snes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/snes_public_gdb.gdb", driver="OpenFileGDB", layer="SNES_Public")

# Set the necessary columns as index
cols = [i for i in snes.columns if i.startswith("TAXON")] + ["PRESENCE_CATEGORY"]


# Function to rasterise the data, note here converting the rasterised data to boolean
def rasterize(geom):
    arr = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=ref_shape,
        transform=ref_transform,
        all_touched=True,
        dtype='uint8',
    )
    
    return arr.astype(np.bool_)


# Rasterise the SNES data
tasks = [delayed(rasterize)(geom) for geom in snes["geometry"]]


# Get the rasterised data
pbar = tqdm(total=len(tasks))

raster_arr = []
for out in tqdm(Parallel(n_jobs=30, return_as='generator')(tasks)):
    raster_arr.append(out)
    pbar.update(1)
    
    
# Save the rasterised data
snes_index_dict = {i: ('idx', snes[i].values) for i in cols}


raster_arr_xr = xarray.DataArray(
    np.array(raster_arr),
    dims=('idx', 'y', 'x'),
    coords={
        'idx': range(len(raster_arr)),
        'y': ref.y, 
        'x': ref.x,
        **snes_index_dict},
)









