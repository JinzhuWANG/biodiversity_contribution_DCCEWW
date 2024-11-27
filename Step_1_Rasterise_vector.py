import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.features
import rioxarray as rxr
import xarray
import sparse
import pickle 
import gzip  # Add this import

from joblib import Parallel, delayed
from tqdm.auto import tqdm


# Set parameters
N_CORES = 50
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": np.bool_}} 

# Read the reference raster data
ref = rxr.open_rasterio("data/NLUM_2010-11_mask.tif", chunks=True)

ref_transform = ref.rio.transform()
ref_crs = ref.rio.crs
ref_shape = ref.rio.shape


# Read the SNES biodiversity data
snes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/snes_public_gdb.gdb", driver="OpenFileGDB", layer="SNES_Public")
ecnes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/ECnes_public_gdb.gdb", driver="OpenFileGDB", layer="ECnes_public")





# Set the necessary columns as index
snes_cols = ['THREATENED_STATUS', 'MIGRATORY_STATUS', 'PRESENCE_RANK', 'PRESENCE_CATEGORY', 'TAXON_GROUP', 
             'TAXON_FAMILY', 'TAXON_ORDER', 'TAXON_CLASS', 'TAXON_PHYLUM', 'TAXON_KINGDOM', 'SCIENTIFIC_NAME']
ecnes_cols = ['COMMUNITY', 'EPBC', 'PRES_RANK', 'CATEGORY', 'REGIONS']



# Function to rasterise the data, note here converting the rasterised data to boolean
def rasterize(geom):
    arr = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=ref_shape,
        transform=ref_transform,
        all_touched=True,
        dtype='uint8',
    )
    
    # Convert to boolean, save as sparse array to save memory
    return sparse.COO.from_numpy(arr.astype(bool))




#######################################################################
# Rasterise the SNES data
#######################################################################
tasks = [delayed(rasterize)(geom) for geom in snes["geometry"]]

raster_arr = []
for out in tqdm(Parallel(n_jobs=N_CORES, return_as='generator')(tasks), total=len(tasks)):
    raster_arr.append(out)

# Convert to xarray
snes_index_dict = {i: ('idx', snes[i].values) for i in snes_cols}

raster_arr_xr = xarray.DataArray(
    sparse.stack(raster_arr, axis=0),  # Use sparse stack to keep the array sparse
    dims=('idx', 'y', 'x'),
    coords={
        'idx': range(len(raster_arr)),
        'y': ref.y, 
        'x': ref.x,
        **snes_index_dict}
)


# Save to Zarr
raster_arr_xr.rio.set_crs(ref_crs)

# Save to gzip compressed pickle
with gzip.open('output/DCCEEW_SNES.pkl.gz', 'wb', compresslevel=9) as f:
    pickle.dump(raster_arr_xr, f)


#######################################################################
# Rasterise the ECNES data
#######################################################################
tasks = [delayed(rasterize)(geom) for geom in ecnes["geometry"]]

raster_arr = []
for out in tqdm(Parallel(n_jobs=N_CORES, return_as='generator')(tasks), total=len(tasks)):
    raster_arr.append(out)
    
# Convert to xarray
ecnes_index_dict = {i: ('idx', ecnes[i].values) for i in ecnes_cols}

raster_arr_xr_ecnes = xarray.DataArray(
    sparse.stack(raster_arr, axis=0),  # Use sparse stack to keep the array sparse
    dims=('idx', 'y', 'x'),
    coords={
        'idx': range(len(raster_arr)),
        'y': ref.y, 
        'x': ref.x,
        **ecnes_index_dict},
)

# Save to Zarr
raster_arr_xr_ecnes.rio.set_crs(ref_crs)
raster_arr_xr_ecnes.name = 'data'

# Save to gzip compressed pickle
with gzip.open('output/DCCEEW_ECNES.pkl.gz', 'wb', compresslevel=9) as f:
    pickle.dump(raster_arr_xr_ecnes, f)






