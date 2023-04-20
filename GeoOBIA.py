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
class GeoOBIA():
    
    def __init__(self, xml_location = None, filePath = None):
        

        """
        Either xml_location or filePath must be provided
        xml_location: path to the xml file
        filePath: path to the folder containing the bands

        """
    

        if xml_location is not None:

            if not os.path.exists(xml_location):
                raise FileNotFoundError(f"File {xml_location} does not exist")
            self.xml_location = xml_location
            
            self.filePath = os.path.split(xml_location)[0]
            self.from_file_path = False
        
        if filePath is not None:
            self.filePath = filePath
            self.from_file_path = True

    def _getProductID(self):
        

        tree = ET.parse(self.xml_location)
        root = tree.getroot()
        xmlns = root.tag[:root.tag.find("}") + 1]

        query = f".//{xmlns}tile_metadata//{xmlns}global_metadata//{xmlns}product_id"
        productID = root.findall(query)[0].text
        return productID
  

    def _getSurfaceReflectance(self):
        
        tree = ET.parse(self.xml_location)
        root = tree.getroot()
        xmlns = root.tag[:root.tag.find("}") + 1]

        query = f".//{xmlns}tile_metadata//{xmlns}band[@product='sr_refl']"
        SRBand = root.findall(query)
        return SRBand
    
    #TODO get other types of bands

    def _getBandName(self):

        if not self.from_file_path:
            BandsNode = self._getSurfaceReflectance()
            Bands_names = [band.attrib["name"] for band in BandsNode]
            return Bands_names

        else:
          
            filePath = self.filePath
            Bands_names = [f.split(".")[0] for f in os.listdir(filePath) if f.endswith(".tif")]
            return Bands_names

    def _getBandPath(self):
       
        if not self.from_file_path: 
            Bands_names = self._getBandName()
            Bands_paths = [os.path.join(self.filePath, f"{self._getProductID()}_{band}.tif") for band in Bands_names]


            return Bands_paths
        else:
            Bands_names = self._getBandName()
            filePath = self.filePath
            Bands_paths = [os.path.join(filePath, f) for f in os.listdir(filePath) if f.endswith(".tif")]


            return Bands_paths

    def _loadBands(self, path):
       
        

        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        with rasterio.open(path) as src:
            
            band = src.read(masked=False)
        return band

    def _getMask(self):

        try:
            BandPath = self._getBandPath()
        except AttributeError:
            BandPath = self._getBandPath()

        with rasterio.open(BandPath[0]) as src:
            mask = src.read_masks(1)
            self.ImgMask = mask
 

    def LoadBands(self):
        
        Bands_paths = self._getBandPath()
        BandName = self._getBandName()

        Bands = {BandName[i]: self._loadBands(Bands_paths[i]).squeeze() for i in range(len(Bands_paths))}


        bandsPathsDict = {BandName[i]: Bands_paths[i] for i in range(len(Bands_paths))}

        self.BandsPath = bandsPathsDict
        self.Bands = Bands
    


    def _reproject(self, srcPath, dstPath, dst_crs):
   
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        with rasterio.open(srcPath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
            'crs': dst_crs,
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
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                


    def Reproject(self, bands , dst_crs, dstPath=None):
        """
        Reproject the bands to a new projection
        bands: list of bands to reproject
        dst_crs: destination crs.  Format = "EPSG:xxxx"
        dstPath: destination path for the reprojected bands
               
        
        """


        BandsPaths = []
        for b in bands:
            if b not in self.BandsPath.keys():
                raise ValueError(f"Band {b} does not exist")
            BandsPaths.append(self.BandsPath[b])

        BandsName = bands


        if dstPath is None:
            return "Please provide the destination path for the bands"

      
        Bands_dst_paths = [os.path.join(dstPath, f"Reproj_{bn}.tif") for bn in BandsName]

        for i in range(len(BandsPaths)):
            self._reproject(BandsPaths[i], Bands_dst_paths[i], dst_crs)


        self.reproj = True
        self._getBandPath = Bands_dst_paths

        bandsPathsDict = {BandsName[i]: Bands_dst_paths[i] for i in range(len(Bands_dst_paths))}

        self.BandsPath = bandsPathsDict

        Bands = {BandsName[i]: self._loadBands(Bands_dst_paths[i]).squeeze() for i in range(len(Bands_dst_paths))}
        self.Bands = Bands





    def _ClipBand(self, bandPath, clipMaskPathShp, dstPath):

        import rasterio.mask
      

        if not os.path.exists(clipMaskPathShp):
            raise FileNotFoundError(f"File {clipMaskPathShp} does not exist")
        if not os.path.exists(bandPath):
            raise FileNotFoundError(f"File {bandPath} does not exist")
        
        maskShp = gpd.read_file(clipMaskPathShp)
        with rasterio.open(bandPath) as src:
            out_image, out_transform = rasterio.mask.mask(src, maskShp.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})
            with rasterio.open(dstPath, "w", **out_meta) as dest:
                dest.write(out_image)

    
    def ClipBands(self, Bands, clipMaskPathShp, dstPaths=None):

        """
        Clip the bands to a shapefile
        Bands: list of bands to clip
        clipMaskPathShp: path to the shapefile
        dstPaths: destination path for the clipped bands
              
        
        """

       
        if dstPaths is None:
            return "Please provide the destination path for the bands"
       
        BandsDstPath = []
        for b in Bands:
            if b not in self.BandsPath.keys():
                raise ValueError(f"Band {b} does not exist")
            
            BandsDstPath.append(os.path.join(dstPaths, f"Clipped_{b}.tif"))
        
        BandsName = Bands

        for i in range(len(BandsDstPath)):
            bandPath = self.BandsPath[BandsName[i]]
            self._ClipBand(bandPath, clipMaskPathShp, BandsDstPath[i])

        self._getBandPath = BandsDstPath
        Bands = {BandsName[i]: self._loadBands(BandsDstPath[i]).squeeze() for i in range(len(BandsDstPath))}

        bandsPathsDict = {BandsName[i]: BandsDstPath[i] for i in range(len(BandsDstPath))}
        self.BandsPath = bandsPathsDict

        self.Bands = Bands


    def _NormalizeData(self, data):
            
            return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    def Segment(self, *bands, **kwargs):

        """
        For now only felzenszwalb algorithm is implemented
        args: bands: the bands to be used for segmentation
        kwargs: the parameters for the segmentation algorithm
          
        
        """

        try:
            self.Bands
        except AttributeError:
            self.LoadBands()

        from skimage.segmentation import felzenszwalb
        import numpy as np

        
        dat = np.array([self._NormalizeData(self.Bands[band]) for band in bands]).transpose([1, 2, 0])

        SegRes = felzenszwalb(dat, **kwargs, channel_axis=-1)
        self.SegRaster = SegRes
    

    def _getGeoProfile(self):
     
        bandPath = self._getBandPath()[0]
        with rasterio.open(bandPath) as src:
            profile = src.profile
        self.geoProfile = profile

    def _SaveArray_GeoTiff(self, array, profileUpdate = None, dstPath = None):

        
        try:
            profile = self.geoProfile
        
        except AttributeError:
            self._getGeoProfile()
            profile = self.geoProfile


        if dstPath is None:
            return "Please provide the destination path for the segmentation result"

        if profileUpdate is not None:
            profile.update(profileUpdate)
        with rasterio.open(dstPath, 'w', **profile) as dst:
            dst.write(array, 1)


    def SaveSegmentation(self, SegPath=None, **kwargs):
            if SegPath is None:
                return "Please provide the destination path for the segmentation result"
            try:
                self.SegRaster
            except AttributeError:
                return "Please run the segmentation first"

            else:
                self._SaveArray_GeoTiff(self.SegRaster, **kwargs, dstPath = SegPath)
                self.SegPathRaster = SegPath

    def Polygonize(self, dstPath=None):
        import rasterio.features
  
        from shapely.geometry import shape

        if dstPath is None:
            return "Please provide the destination path for the segmentation result"
        try:
            self.SegPathRaster  
        except AttributeError:
            return "Please run the segmentation first"
        
        try:
            self.ImgMask
        except AttributeError:
            self._getMask()

        with rasterio.open(self.SegPathRaster) as src:
            SegResRaster = src.read(1)
            crs = src.crs
            Poly = []
            for geom, val in rasterio.features.shapes(SegResRaster, transform=src.transform, mask=self.ImgMask):
                polygone = {"geometry": shape(geom), "properties": {"value": val}}
                Poly.append(polygone)
        
        self.crs = crs
        SegVec = gpd.GeoDataFrame.from_features(Poly).dissolve(by="value")
        SegVec.to_file(dstPath, driver="ESRI Shapefile", crs=self.crs)
        self.SegVec = SegVec


    def ComputeGeoAttri(self):
        

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
            self.SegVec[att] = ShapeStats.__getattribute__(att)(self.SegVec.geometry)

    def _ComputeBandsAttri(self, band, Stats):
    
        

        
        try:
            self.SegRaster
        except AttributeError:
            return "Please run the segmentation first"
        
        try:
            Img = self.Bands.get(band)
        except AttributeError:
            return "Please load the bands first"
        df = pd.DataFrame(Img.flatten(), columns=[band], index=self.SegRaster.flatten())
        df = df.groupby(df.index).agg(Stats)
        df.columns = [f"{band}_{stat}" for stat in Stats]   

        return df
    
    def ComputeBandsAttri(self, Bands, Stats = ['mean', 'max', 'min', 'std', 'median']):
    
        
        df = self._ComputeBandsAttri(Bands[0], Stats)
        for band in Bands[1:]:
            df = df.merge(self._ComputeBandsAttri(band, Stats), left_index=True, right_index=True, how='left')

        self.SegVec = self.SegVec.merge(df, left_on='value', right_index=True, how='left')

    def LoadTrainingSamples(self, TrainingPath):
      
        TrainingSamples = gpd.read_file(TrainingPath)

        with rasterio.open(self.SegPathRaster) as src:
            coor_list = [(x, y) for x, y in zip(TrainingSamples.geometry.x, TrainingSamples.geometry.y)]
            TrainingSamples['value'] = [float(val[0]) for val in src.sample(coor_list)]

        self.TrainingSamples = (self.SegVec
                                .merge(TrainingSamples, left_on='value', right_on='value', how='right')
                                       .drop_duplicates('value') #duplicate values are removed
                                                        )
    
    def TrainClassifier(self, clf, Xvars, Yvar):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
     

        try:

            X = self.TrainingSamples[Xvars]
            y = self.TrainingSamples[Yvar]
        except AttributeError:
            return("Please load the training samples first")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ClassificationResults = {'Accuracy': accuracy_score(y_test, y_pred),
                                                            'Precision': precision_score(y_test, y_pred, average='weighted'),
                                                              'Recall': recall_score(y_test, y_pred, average='weighted'),
                                                                                     'F1': f1_score(y_test, y_pred, average='weighted'),
                                                                                                    'Confusion Matrix': confusion_matrix(y_test, y_pred),
                                                                                                                                         'Classification Report': classification_report(y_test, y_pred)}
        self.Xvars = Xvars
        self.Yvar = Yvar
        self.clf = clf   
        self.ClassificationResults = ClassificationResults



    def Predict(self):

        try:
            clf = self.clf
        except AttributeError:
            return("Please train the classifier first")
        
        Y = clf.predict(self.SegVec[self.Xvars])
        self.SegVec[self.Yvar] = Y

    def ExportPredictedVec(self, dstPath=None):

        if dstPath is None:
            return "Please provide the destination path for the segmentation result"
        try:
            self.SegVec
        except AttributeError:
            return "Please run the vectorization (Polygonize) first"
        
        self.SegVec.to_file(dstPath, driver="ESRI Shapefile", crs=self.crs)

    #TODO: Export the predicted raster
    def ExportPredictedRaster(self, dstPath=None):

        if dstPath is None:
            return "Please provide the destination path for the segmentation result"
        try:
            self.SegVec
        except AttributeError:
            return "Please run the vectorization (Polygonize) first"
        
