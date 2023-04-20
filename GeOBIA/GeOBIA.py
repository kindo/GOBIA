#Python version: 3.10.8
#geopandas version: 0.12.2
#rasterio version: 1.2.10
#gdal version: 3.5.1
#skimage version: 0.19.3 
#sklearn version: 1.2.0

from . import ShapeStats
import geopandas as gpd
import os
from xml.etree import ElementTree as ET
import rasterio
import numpy as np
import pandas as pd
import re
import warnings

class GeoOBIA():
    
    def __init__(self, filePath = None):
        

        """

        filePath: path to the folder containing the bands

        """
        if filePath is None:
            self.filePath = os.getcwd()
            self._getBandName()
        
        else:
            self.filePath = filePath
            self._getBandName()

    def _getBandName(self):

        regx = re.compile("B[0-9].TIF|B10.TIF|B11.TIF")
        getdate = lambda f: "_".join(os.path.split(f)[-1].split('_')[3:5])
        getlandsatNumb = lambda f: os.path.split(f)[-1].split('_')[0]
        getbname = lambda f: regx.findall(f.upper())[0].split(".")[0]

        if isinstance(self.filePath, list):
            filesPathList = []
            for p in self.filePath:
                if not os.path.exists(p):
                    raise FileNotFoundError("File not found")
                else:
                    for f in os.listdir(p):
                        if regx.findall(f.upper()) != []:
                            filesPathList.append(os.path.join(p, f))

            
            
            
            bandsNamePath = {f"{getlandsatNumb(f)}_{getdate(f)}_{getbname(f)}": f for f in filesPathList}
        
        elif isinstance(self.filePath, str):


            if not os.path.exists(self.filePath):
                raise FileNotFoundError("File not found")
        
            files = os.listdir(self.filePath)
        
            filesPathList = [os.path.join(self.filePath, f) for f in files if regx.findall(f.upper()) != []]

            bandsNamePath = {f"{getlandsatNumb(f)}_{getdate(f)}_{getbname(f)}": f for f in filesPathList}
        else:
            raise TypeError("filePath must be a string or a list of string")

        self.Bands = list(bandsNamePath.keys())
        self.BandsNamePath = bandsNamePath
        self.Geoinfo = self._getGeoInfo(self.Bands[0]) #TODO : check if all bands have the same geoinfo

    def _getBound(self, band):
        if band in self.Bands:
            path = self.BandsNamePath[band]
        else:
            return FileExistsError("Band not found")
       
        with rasterio.open(path) as src:
            bound = src.bounds
            return bound

    def _loadBands(self, band):
        
        if band in self.Bands:
            path = self.BandsNamePath[band]
        else:
            return FileExistsError("Band not found")
       
        with rasterio.open(path) as src:
            
            bound = src.bounds
            bandArray = src.read(masked=False)
        return bandArray.squeeze()

    def _getMask(self):

        try:
            BandPath = list(self.BandsNamePath.values())
        except AttributeError:
            self._getBandName()
            BandPath = list(self.BandsNamePath.values())

        with rasterio.open(BandPath[0]) as src:
            mask = src.read_masks(1)
            profile = src.profile
            self.geoProfile = profile
            self.ImgMask = mask
 
    def LoadBands(self, bands = None):

        """
        Load the bands into a dictionary
        bands: list of bands to load. If None, all bands in filePath will be loaded
        
        """

        if bands is None:
            try:
                bands = self.Bands
            except AttributeError:
                self._getBandName()
                bands = self.Bands
        
        Bounds = {b: self._getBound(b) for b in bands}
        BandsArray = {b: self._loadBands(b) for b in bands}


        bandsNamePath = {b: self.BandsNamePath[b] for b in bands}
        self.BandsNamePath = bandsNamePath
        self.Bands = list(self.BandsNamePath.keys())
        self.BandsArray = BandsArray
        self.Bounds = Bounds
        self._getMask()
        
        
        return "Successfully loaded bands. Call self.BandsArray to get the bands in np.array format"

    def _getGeoInfo(self, band):
        if band in self.Bands:
            path = self.BandsNamePath[band]
        else:
            return FileExistsError("Band not found")
       
        with rasterio.open(path) as src:
            bound = src.bounds
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height


            return {"bound" : bound, "crs" : crs, "transform" : transform, "width" : width, "height" : height}

    def _reproject(self, band, dstPath, crs):
   
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        srcPath = self.BandsNamePath[band]

        with rasterio.open(srcPath) as src:

            bounds, scrs, transform, width, height = tuple(self.Geoinfo.values())
            #transform, width, height = calculate_default_transform(
                #src.crs, crs, src.width, src.height, *src.bounds)

            transform, width, height = calculate_default_transform(
                scrs, crs, width, height, *bounds) #if dont work to delete
            
            kwargs = src.meta.copy()
            kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
            })
            
            with rasterio.open(dstPath, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest)
                
    def Reproject(self, crs = None, bands = None , dstPath=None, reloadBands=False):
        """
        Reproject the bands to a new projection
        bands: list of bands to reproject
        crs: destination crs.  Format = "EPSG:xxxx"
        dstPath: destination path for the reprojected bands. The path should be a folder.
               
        
        """

        if crs is None:
            return "Please provide the destination crs"

        if bands is None:
            bands = self.Bands
        else:
            for b in bands:
                if b not in self.Bands:
                    raise ValueError(f"Band {b} not found")
            

        
        if dstPath is None:
            return "Please provide the destination path for the bands"

      
        bdstPath = [os.path.join(dstPath, f"Reproj_{b}.tif") for b in bands]



        for i in range(len(bdstPath)):
            self._reproject(bands[i], bdstPath[i], crs)

        bandsNamePath = {bands[i]: bdstPath[i] for i in range(len(bdstPath))}


        #Reload the BandArray
        self.filePath = dstPath
        self.Bands = list(bandsNamePath.keys())
        self.BandsNamePath = bandsNamePath
        self.Bounds = self._getBound(self.Bands[0])
        self.reproj = True
        self.Geoinfo = self._getGeoInfo(self.Bands[0])

        if reloadBands:
            self.LoadBands()

        return "Successfully reprojected bands"
        
    def _ClipBand(self, band, clipMaskPathShp, dstPath):

        import rasterio.mask
      
        bandPath = self.BandsNamePath[band]
        
        if not os.path.exists(clipMaskPathShp):
            raise FileNotFoundError(f"File {clipMaskPathShp} does not exist")
        if not os.path.exists(bandPath):
            raise FileNotFoundError(f"File {bandPath} does not exist")
        
        maskShp = gpd.read_file(clipMaskPathShp)

        mEPSG = maskShp.crs.to_epsg()
        bEPSG = self.Geoinfo["crs"].to_epsg()

        if mEPSG != bEPSG:
            print("Reprojecting the mask to the bands crs")
            maskShp = maskShp.to_crs(self.Geoinfo["crs"])

        with rasterio.open(bandPath) as src:
            out_image, out_transform = rasterio.mask.mask(src, maskShp.geometry, crop=True, all_touched = True, pad_width=0)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})
            with rasterio.open(dstPath, "w", **out_meta) as dest:
                dest.write(out_image)

    def ClipBands(self, clipMaskPathShp, dstPath, bands=None, reloadBands=False):


        """
        Clip the bands to a shapefile
        Bands: list of bands to clip
        clipMaskPathShp: path to the mask shapefile
        dstPath: destination path for the clipped bands
              
        
        """

        if bands is None:
            bands = self.Bands
        else:
            for b in bands:
                if b not in self.Bands:
                    raise ValueError(f"Band {b} not found")
            

        


      
        bdstPath = [os.path.join(dstPath, f"Clipped_{b}.tif") for b in bands]

        for i in range(len(bdstPath)):
            self._ClipBand(bands[i], clipMaskPathShp, bdstPath[i])

        bandsNamePath = {bands[i]: bdstPath[i] for i in range(len(bdstPath))}

        self.filePath = dstPath
   
        self.Bands = list(bandsNamePath.keys())
        self.BandsNamePath = bandsNamePath
        self.Bounds = self._getBound(self.Bands[0])
        self.Geoinfo = self._getGeoInfo(self.Bands[0])

        if reloadBands:
            self.LoadBands()
        
        return "Successfully clipped bands"
    
    def _loadDEM(self, demPath):

        if os.path.exists(demPath):
            with rasterio.open(demPath) as src:
                dem = src.read(1)
                dem = dem.astype(np.float32)
            
                return dem
        else:
            raise FileNotFoundError(f"File {demPath} does not exist")

    def AddDEM(self, were_save):


        """
        The function downloads the ALOS DEM and preprocess it to be used in the analysis
        were_save: path to the folder where the DEM will be saved
        
        """

        from .DownloadAlosDem import DownloadALOSDEM, ProcessDEMs
        import pyproj

        try:
            src_bounds = self.Geoinfo['bound']
        except AttributeError:
            return "Please load the bands first"

        if not os.path.exists(were_save):
            os.makedirs(were_save)

        crsDst = pyproj.CRS.from_epsg(4326)
        crsSrc = pyproj.CRS.from_epsg(self.Geoinfo["crs"].to_epsg())

        
        transformer = pyproj.Transformer.from_crs(crsSrc, crsDst, always_xy=True)

        MinLong, MinLat = transformer.transform(src_bounds.left, src_bounds.bottom)
        MaxLong, MaxLat = transformer.transform(src_bounds.right, src_bounds.top)

        DEMDownloadPath = DownloadALOSDEM(MinLat = MinLat, MaxLat = MaxLat, MinLong = MinLong, 
                    MaxLong = MaxLong,  SavePath = were_save)

        ExamplePath = self.BandsNamePath[self.Bands[0]]
        files_to_delete, DEMPath = ProcessDEMs(were_save, ExamplePath, DEMDownloadPath)
        
        
        for f in files_to_delete:
            os.remove(os.path.join(were_save, f))

        demName = os.path.split(DEMPath)[1].split(".")[0]
        
        self.Bands.append(demName)
        self.BandsNamePath[demName] = DEMPath
        try:
            self.BandsArray[demName] = self._loadDEM(DEMPath)
        except AttributeError:
            self.LoadBands()
            self.BandsArray[demName] = self._loadDEM(DEMPath)

    @staticmethod
    def NormalizeData(data):
            
            return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    def Segment(self, bands = None, **kwargs):

        """
        For now only felzenszwalb algorithm is implemented
        args: bands: the bands to be used for segmentation
        kwargs: the parameters for the segmentation algorithm
          
        
        """

        #verify if the bands have the same size
        

        if bands is None:
            bands = self.Bands
        else:
            for b in bands:
                if b not in self.Bands:
                    raise ValueError(f"Band {b} not found")

        self.LoadBands()

        for b in bands:
            if self.BandsArray[b].shape != self.BandsArray[bands[0]].shape:
                raise ValueError("The bands do not have the same size")


        

        from skimage.segmentation import felzenszwalb
        import numpy as np

        
        dat = np.array([self.NormalizeData(self.BandsArray[band]) for band in bands]).transpose([1, 2, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SegRes = felzenszwalb(dat, **kwargs, channel_axis=-1)

        self.SegRaster = SegRes

        return "Segmentation done. Result is stored in the SegRaster attribute of the object"
    
    def _SaveArrayGeoTiff(self, array, profileUpdate = None, dstPath = None):

        #get the context of the raster

        try:
            profile = self.geoProfile
        
        except AttributeError:
            self._getMask()
            profile = self.geoProfile


        if profileUpdate is not None:
            profile.update(profileUpdate)
        with rasterio.open(dstPath, 'w', **profile) as dst:
            dst.write(array, 1)

    def SegSaveRaster(self, SegPath=None, profileUpdate = None):
            
            if SegPath is None:
                return "Please provide the destination path for the segmentation result"
            try:
                self.SegRaster
            except AttributeError:
                return "Please run the segmentation first"

            else:
                self._SaveArrayGeoTiff(self.SegRaster, profileUpdate, dstPath = SegPath)
                self.SegPathRaster = SegPath

    def VectorizeSegRes(self, dstPath=None):

        """
        Vectorize the segmentation result
        dstPath: destination path for the vectorized segmentation result. If None, the result is saved locally
        The result is stored in the attribute SegVector
        
        """

        import rasterio.features
        from shapely.geometry import shape

        
        try:
            self.SegRaster  
        except AttributeError:
            return "Please run the segmentation first"
        
        try:
            self.ImgMask
        except AttributeError:
            self._getMask()

        try:
            BandPath = list(self.BandsNamePath.values())
        except AttributeError:
            self._getBandName()
            BandPath = list(self.BandsNamePath.values())

        with rasterio.open(BandPath[0]) as src:
            crs = src.crs
            Poly = []
            for geom, val in rasterio.features.shapes(self.SegRaster.astype("int32"), transform=src.transform, mask=self.ImgMask):
                polygone = {"geometry": shape(geom), "properties": {"value": val}}
                Poly.append(polygone)
        
        self.crs = crs
        SegVec = gpd.GeoDataFrame.from_features(Poly).dissolve(by="value")

        if dstPath is not None:
           
            SegVec.to_file(dstPath, driver="ESRI Shapefile", crs=self.crs)
        
        self.SegVector = SegVec

   

        return "Vectorization done. Result is stored in the SegVector attribute of the object"

    def ExportVector(self, Vector = None, dstPath=None):

        if dstPath is None:
                return "Please provide the destination path for the vectorized segmentation result"

        if Vector is None:
        
            
            try:
                self.SegVector
            except AttributeError:
                return "Please run the vectorization first"

            else:
                self.SegVector.to_file(dstPath, driver="ESRI Shapefile", crs=self.crs)
            self.SegPathVector = dstPath
        
        else:
            if isinstance(Vector, gpd.GeoDataFrame):
                try:
                    Vector.crs
                except AttributeError:
                    return "Please provide a vector with a crs"
                else:
                    Vector.to_file(dstPath, driver="ESRI Shapefile", crs=Vector.crs)

    def ComputeGeoAttri(self):
        
        """
        Compute the geometric attributes of the segmentation result

        The attributes available are:
        boundary_amplitude, convex_hull_ratio, diameter_ratio, 
        equivalent_rectangular_index, fractal_dimension, isoarea_quotien, 
        isoperimetric_quotien, length_width_ratio, minimum_bounding_circle, 
        radii_ratio, rectangularity, shape_index, squareness
        
        """

        GeoAtt = ['boundary_amplitude',
                  'convex_hull_ratio',
                  'diameter_ratio',
                    'equivalent_rectangular_index',
                    'fractal_dimension',
                    'isoarea_quotien',
                    'isoperimetric_quotien',
                    'length_width_ratio',
                    'minimum_bounding_circle',
                    'radii_ratio',
                    'rectangularity',
                    'shape_index',
                    'squareness',]
        
        for att in GeoAtt:
            self.SegVector[att] = ShapeStats.__getattribute__(att)(self.SegVector.geometry)

        return "Geometric attributes computed. Result is stored in the SegVector attribute of the object"

    def _ComputeBandsAttri(self, band, Stats):
    
        
        if band not in self.BandsArray.keys():
            FileNotFoundError(f"{band} is not in the loaded bands")
        
        try:
            self.SegRaster
        except AttributeError:
            return "Please run the segmentation first"
        
        
        Img = self.BandsArray.get(band)
        
        
        df = pd.DataFrame(Img.flatten(), columns=[band], index=self.SegRaster.flatten())
        df = df.groupby(df.index).agg(Stats)
        df.columns = [f"{band}_{stat}" for stat in Stats]   

        return df
    
    def ComputeBandsAttri(self, Bands = None, Stats = ['mean', 'max', 'min', 'std', 'median']):
    
        if Bands is None:
            Bands = self.Bands
        else:
            for band in Bands:
                if band not in self.Bands:
                    return f"{band} is not in the loaded bands"

        try:
            self.SegVector
        except AttributeError:
            return "Please run the vectorization first: 'VectorizeSegRes'"

        df = self._ComputeBandsAttri(Bands[0], Stats)
        for band in Bands[1:]:
            df = df.merge(self._ComputeBandsAttri(band, Stats), left_index=True, right_index=True, how='left')

        
        self.SegVector = self.SegVector.merge(df, left_on='value', right_index=True, how='left')

    def LoadTrainingSamples(self, TrainingPath):
      
        TrainingSamples = gpd.read_file(TrainingPath)

        with rasterio.Env():

            try:
                segRasterPath = self.SegPathRaster
            except AttributeError:
                segRasterPath = "temp.tif"
                self.SegSaveRaster(segRasterPath)
                flag = True

            with rasterio.open(segRasterPath, "r") as src:

                coor_list = [(x, y) for x, y in zip(TrainingSamples.geometry.x, TrainingSamples.geometry.y)]
                TrainingSamples['value'] = [float(val[0]) for val in src.sample(coor_list)]
            
            if flag:
                os.remove("temp.tif")
                delattr(self, "SegPathRaster")


        self.TrainingSamples = (self.SegVector
                                .merge(TrainingSamples, left_on='value', right_on='value', how='right')
                                       .drop_duplicates('value') #duplicate values are removed
                        
                                                        )

        
        return "Training samples loaded. Result is stored in the TrainingSamples attribute of the object"
    
    def TrainClassifier(self, clf, Xvars, Yvar, scaler=None):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
     

        try:
     
            self.TrainingSamples = self.TrainingSamples.dropna(subset=[*Xvars, Yvar])
            X = self.TrainingSamples[Xvars]
            y = self.TrainingSamples[Yvar]
        except AttributeError:
            return("Please load the training samples first")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)

        if scaler is not None:
            scaleX = scaler()
            X_train = scaleX.fit_transform(X_train)
            X_test = scaleX.transform(X_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ClassificationResults = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Classification Report': classification_report(y_test, y_pred)
                                                           }
        
        self.Xvars = Xvars
        self.Yvar = Yvar
        if scaler is not None:
            self.scaler = scaleX
   
        self.clf = clf   
        self.ClassificationResults = ClassificationResults

    def Predict(self, savePath = None):

        """
        Predict the class of each segment and store the result in the SegVector attribute of the object
        If a path is provided, the result is exported as a shapefile
        
        savePath: path to the destination shapefile
        
        """

        try:
            clf = self.clf
        except AttributeError:
            return "Please train the classifier first"
        
        X_data = self.SegVector[self.Xvars]

        try :
            scaleX = self.scaler
            X_data = scaleX.transform(X_data)
        except AttributeError:
            pass

        Y = clf.predict(X_data)
        self.SegVector[self.Yvar] = Y
        self.Predicted = True

        if savePath is not None:
            self._exportPredictedVec(savePath)

    def _exportPredictedVec(self, dstPath=None):

        if dstPath is None:
            return "Please provide the destination path for the segmentation result"
        
        if not self.Predicted:
            return "Please run the prediction first"
        
        self.SegVector.dissolve(self.Yvar).to_file(dstPath, driver="ESRI Shapefile", crs=self.crs)


        
