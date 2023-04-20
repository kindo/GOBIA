def DownloadALOSDEM(MinLat:int, MaxLat:int, MinLong:int, 
                    MaxLong:int,  SavePath:str) -> None:
    
    """
    
    This is a convenient function to download ALOS DEM from opentopography, 
    
    Long Lat are in WGS84 (epsg:4326)
    

    
    The following Packages are required boto3 to connect to AWS S3 bucket, botocore, numpy, os and pandas
    
    """
    
    import boto3
    from botocore.config import Config
    import numpy as np
    from botocore import UNSIGNED
    import numpy as np
    import pandas as pd
    import os
    s3 = (boto3
          .resource('s3',endpoint_url='https://opentopography.s3.sdsc.edu', 
                    config=Config(signature_version=UNSIGNED)))
    
    bucket = s3.Bucket("raster")
    
    #Create a list of all the ALOS tiles available 
    List_ALOSDEM = []

    for obj in bucket.objects.filter(Prefix="AW3D30/AW3D30_global/ALPSMLC30").all():
        List_ALOSDEM.append(obj.key)
        

    Easting = {'W':-1, 'E': 1}
    Northing = {'S': -1, 'N': 1}

    DEMS = (pd.DataFrame({'List_ALOSDEM':List_ALOSDEM}).assign(Lat = lambda df: df.List_ALOSDEM.str[32:35], 
                                                  Long = lambda df: df.List_ALOSDEM.str[36:39],
                                                EASTING = lambda df: df.List_ALOSDEM.str[35],
                                            NORTHING = lambda df: df.List_ALOSDEM.str[31])
                                          .assign(LongE = lambda df: df.Long.astype('int') * df.EASTING.map(Easting),
                                                  LatN = lambda df: df.Lat.astype('int') * df.NORTHING.map(Northing))
    )
    
    MinLong, MinLat, MaxLong, MaxLat = np.floor(MinLong), np.floor(MinLat), np.ceil(MaxLong), np.ceil(MaxLat)

    DEM_Download = DEMS.loc[DEMS.LongE.between(MinLong, MaxLong) & DEMS.LatN.between(MinLat, MaxLat), 'List_ALOSDEM'] 
    print(f"The number of DEM that will be download is: {len(DEM_Download)}")
    
    DEMPath = []
    for i in range(DEM_Download.shape[0]):
    
        SPath = os.path.join(os.path.normpath(SavePath), DEM_Download.iloc[i][31:])
        DEMPath.append(SPath)
        bucket.download_file(DEM_Download.iloc[i], SPath)
        print(f"DEM {i+1} downloaded")
    
    return DEMPath


def ProcessDEMs(SavePath, RasterExamplePath, DEMsPaths =None):

    """
    This function will merge all the DEMs in the SavePath folder, reproject and clip them to the RasterExamplePath
    
    SavePath: Path to the folder containing the DEMs to merge (only non tif will be ignored)
    RasterExamplePath: Path to the example raster to clip the DEMs to
    
    
    """

    #TODO use only the files in the folder and not all the tif files in the folder

    from rasterio.io import MemoryFile
    import os
    from rasterio.merge import merge
    import rasterio.mask
    import rasterio
    import geopandas as gpd
    from rasterio.warp import calculate_default_transform, reproject, Resampling
 
    def vectorize(src):
        from shapely.geometry import shape
        Poly = []
        for geom, val in rasterio.features.shapes(src.dataset_mask().astype('int'), transform=src.transform, mask=src.dataset_mask()):
            polygone = {"geometry": shape(geom), "properties": {"value": val}}
            Poly.append(polygone)
        return gpd.GeoDataFrame.from_features(Poly).dissolve(by="value")

    with rasterio.Env():

        #list of files to merge SavePath must contain only the files to merge

        if DEMsPaths is None:
            files = os.listdir(os.path.normpath(SavePath))
        else:
            files = [p for p in DEMsPaths if os.path.exists(p)]
        files_merge = []

        fnames  = []

        for f in files:
            #check if the file is a tif
            if f.split(".")[-1] != "tif":
                continue
            if DEMsPaths is None:
                file = os.path.join(os.path.normpath(SavePath), f)
            else:
                file = f
            fnames.append(file)
            src = rasterio.open(file)
            #take the geoinformation from the example raster and update later
            meta = src.meta
            files_merge.append(src)

        #merge files and update metadata
        mosaic, out_trans = merge(files_merge)
        meta.update({"driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform":out_trans
                        })
        
        #Take the geoinformation from the example raster and create a mask in vector format
        with rasterio.open(os.path.normpath(RasterExamplePath)) as src:
            dst_bound = src.bounds
            dst_crs = src.crs
            dst_transform = src.transform
            dst_width = src.width
            dst_height = src.height

            #create a mask in vector format
            mask = vectorize(src).simplify(0.0001)

        


        #Open the merged file in memory
        with MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(mosaic)
                
                #Compute the transformation to the example raster
                transform, width, height = calculate_default_transform(
                dataset.crs, dst_crs, dataset.width, dataset.height, *dataset.bounds) 

                kwargs = dataset.meta.copy()

                #Update the metadata with the transformation to the example raster
                kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
                })

                with MemoryFile() as memfile2:
                    #Create a new file in memory with the updated metadata (reprojected)
                    with memfile2.open(**kwargs) as dataset_r:
                        reproject(
                        source=rasterio.band(dataset, 1),
                        destination=rasterio.band(dataset_r, 1),
                        src_transform=dataset.transform,
                        src_crs=dataset.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

                        with MemoryFile() as memfile3:
                            #Create a new file in memory to crop the image
                            with memfile3.open(**dataset_r.meta) as dataset_c:
                                out_image, out_transform = rasterio.mask.mask(dataset=dataset_r, shapes=mask.geometry, crop=True)
                                out_meta = dataset_c.meta.copy()
                                out_meta.update({"driver": "GTiff",
                                            
                                             
                                                "height": out_image.shape[1],
                                                "width": out_image.shape[2],
                                          
                                                "transform": out_transform
                                                })
                                with MemoryFile() as memfile4:
                                    with memfile4.open(**out_meta) as dest:
                                        dest.write(out_image)

                                        des_transform, des_width, des_height = calculate_default_transform(src_crs=dest.crs, dst_crs=dest.crs, 
                                        width=dest.width, height=dest.height, 
                                        left=dest.bounds.left, bottom=dest.bounds.bottom, 
                                        right=dest.bounds.right, top=dest.bounds.top,
                                        dst_width=src.width, dst_height=src.height)

                                        out_meta = dest.meta.copy()

                                        out_meta.update({
                                            'transform': des_transform,
                                            'width': des_width,
                                            'height': des_height
                                            })
                                        


                                        with rasterio.open(os.path.join(SavePath, f"mosaic_EPSG{dst_crs.to_epsg()}.tif"), "w", **out_meta) as final_dat:
                                            reproject(
                                                    source=rasterio.band(dest, 1),
                                                    destination=rasterio.band(final_dat, 1),
                                                    src_transform=dest.transform,
                                                    src_crs=dest.crs,
                                                    dst_transform=des_transform,
                                                    dst_crs=src.crs,
                                                    resampling=Resampling.nearest)
    return fnames, os.path.join(SavePath, f"mosaic_EPSG{dst_crs.to_epsg()}.tif")