import os
import gzip
import xarray
import sparse
import pickle 
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.features
import rioxarray as rxr

from joblib import Parallel, delayed
from tqdm.auto import tqdm


# Set parameters
N_CORES = 50
SNES_TIF_PATH = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF'



# Read the reference raster data
ref = rxr.open_rasterio("data/NLUM_2010-11_mask.tif", chunks=True).squeeze('band').drop_vars('band')
ref_mask = ref.values

ref_meta = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'nodata': 255,
    'width': ref.rio.width,
    'height': ref.rio.height,
    'count': 1,
    'crs': ref.rio.crs,
    'transform': ref.rio.transform(),
    'compress': 'lzw',  
}




# Read the SNES biodiversity data
snes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/snes_public_gdb.gdb", driver="OpenFileGDB", layer="SNES_Public")
ecnes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/ECnes_public_gdb.gdb", driver="OpenFileGDB", layer="ECnes_public")

# Set the necessary columns as index
snes_cols = ['THREATENED_STATUS', 'MIGRATORY_STATUS', 'PRESENCE_RANK', 'PRESENCE_CATEGORY', 'TAXON_GROUP', 
             'TAXON_FAMILY', 'TAXON_ORDER', 'TAXON_CLASS', 'TAXON_PHYLUM', 'TAXON_KINGDOM', 'SCIENTIFIC_NAME']
ecnes_cols = ['COM_ID','COMMUNITY', 'EPBC', 'PRES_RANK', 'CATEGORY', 'REGIONS']


# Define the k-v pair for presence 
presence_dict = {1: 'MAYBE', 2: 'LIKELY'}

# Dissolve to merge 
snes_dissolve = snes.dissolve(by=['SCIENTIFIC_NAME','PRESENCE_CATEGORY']).reset_index()
ecnes_dissolve = ecnes.dissolve(by=ecnes_cols).reset_index()

if not os.path.exists(f"{SNES_TIF_PATH}/snes_dissolve.shp"):
    snes_dissolve[snes_cols + ['geometry']].to_file(f"{SNES_TIF_PATH}/DISSOLVED_VECTOR/snes_dissolve.shp")
if not os.path.exists(f"{SNES_TIF_PATH}/ecnes_dissolve.shp"):
    ecnes_dissolve[ecnes_cols + ['geometry']].to_file(f"{SNES_TIF_PATH}/DISSOLVED_VECTOR/ecnes_dissolve.shp")


def get_presVal_savePath(row):
    # Get value for rasterisation polygon (1 for 'maybe present', 2 for 'likely present')
    if 'PRES_RANK' in row:  # ECNES data
        val = row['PRES_RANK']
        name = row['COMMUNITY'].replace('/', '_')
        save_path = rf'{SNES_TIF_PATH}/ECNES/{name}_{presence_dict[val]}.tif'
    else:                   # SNES data
        val = row['PRESENCE_RANK']
        name = row['SCIENTIFIC_NAME'].replace('/', '_')
        save_path = f'{SNES_TIF_PATH}/SNES/{row["TAXON_GROUP"]}/{name}/{name}_{presence_dict[val]}.tif'
    
    # Replace spaces with underscores
    save_path = save_path.replace(' ', '_')
    return val, save_path



# Function to rasterise the data, note here converting the rasterised data to boolean
def rasterize(row):
    val, save_path = get_presVal_savePath(row)
    # Rasterise the polygon
    arr = rasterio.features.rasterize(
        [(row["geometry"], val)],
        out_shape=ref_mask.shape,
        transform=ref_meta['transform'],
        all_touched=False,
        dtype='uint8',
    )
    # Apply mask, 255 will be used for nodata
    arr = np.where(ref_mask, arr, 255)
    # Save to GEOTIFF
    with rasterio.open(save_path, 'w', **ref_meta) as dst:
        dst.write(arr, 1)
        


# Create folders for SNES data
for _,row in snes_dissolve.iterrows():
    tif_path = get_presVal_savePath(row)[1]
    folder = os.path.dirname(tif_path)
    if os.path.exists(folder):
        continue
    os.makedirs(folder, exist_ok=True)
    
# Create folders for ECNES data; Only a single folder to store all the data
if not os.path.exists(f'{SNES_TIF_PATH}/ECNES'):
    os.makedirs(f'{SNES_TIF_PATH}/ECNES', exist_ok=True)
    
    

#######################################################################
# Rasterise and save the SNES data to GEOTIFF
#######################################################################

tasks = [delayed(rasterize)(row) for _,row in snes_dissolve.iterrows()]
for _ in tqdm(Parallel(n_jobs=N_CORES, return_as='generator')(tasks), total=len(tasks)):
    pass

# Save SNES attributes to csv
snes_meta = snes_dissolve[snes_cols].copy()
snes_meta['TIF_PATH'] = snes_meta.apply(lambda x: get_presVal_savePath(x)[1], axis=1)
snes_meta.to_csv(f'{SNES_TIF_PATH}/DCCEEW_SNES_meta.csv', index=False)



#######################################################################
# Rasterise and save the ECNES data to GEOTIFF
#######################################################################
tasks = [delayed(rasterize)(row) for _,row in ecnes_dissolve.iterrows()]

raster_arr = []
for out in tqdm(Parallel(n_jobs=N_CORES, return_as='generator')(tasks), total=len(tasks)):
    raster_arr.append(out)

# Save ECNES attributes to csv
ecnes_meta = ecnes_dissolve[ecnes_cols].copy()
ecnes_meta['TIF_PATH'] = ecnes_meta.apply(lambda x: get_presVal_savePath(x)[1], axis=1)
ecnes_meta.to_csv(f'{SNES_TIF_PATH}/DCCEEW_ECNES_meta.csv', index=False)


