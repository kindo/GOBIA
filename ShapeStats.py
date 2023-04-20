try:
    import shapely
except ImportError:
    print("shapely not installed")

import numpy as np
import pandas as pd


"""
All the code below is based on the code of the package 'esda.shape' 
pygeos was simply replaced with shapely

"""

def _cast(collection):
    return collection
    


def isoperimetric_quotien(collection):
    """Calculate the isoperimetric quotient of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The isoperimetric quotient.
    """
    collection = _cast(collection)
    area = collection.area
    perimeter = collection.length
    return 4 * np.pi * area / perimeter**2

def isoarea_quotien(collection):
    """Calculate the isoarea quotient of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The isoarea quotient.
    """
    collection = _cast(collection)
    area = collection.area
    perimeter = collection.length
    return (2 * np.pi * np.sqrt(area / np.pi)) / perimeter

def minimum_bounding_circle(collection):
    """Calculate the minimum bounding circle of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The radius of the minimum bounding circle.
    """
    collection = _cast(collection)
    mbc = shapely.minimum_bounding_circle(collection)
    return collection.area / mbc.area

def radii_ratio(collection):
    """Calculate the radii ratio of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The radii ratio.
    """
    collection = _cast(collection)
    r_mbc = shapely.minimum_bounding_radius(collection)
    r_eac = np.sqrt(collection.area / np.pi)
    return r_eac / r_mbc

def diameter_ratio(collection, rotated =True):
    """Calculate the diameter ratio of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The diameter ratio.
    """
    ga = _cast(collection)
    if rotated:
        box = shapely.minimum_rotated_rectangle(ga)
        coords = shapely.get_coordinates(box)
        a, b, _, d = (coords[0::5], coords[1::5], coords[2::5], coords[3::5])
        widths = np.sqrt(np.sum((a - b) ** 2, axis=1))
        heights = np.sqrt(np.sum((a - d) ** 2, axis=1))
    else:
        box = shapely.bounds(ga)
        (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
        widths, heights = np.abs(xmax - xmin), np.abs(ymax - ymin)
    return np.minimum(widths, heights) / np.maximum(widths, heights)

def length_width_ratio(collection):
    """Calculate the length-width ratio of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The length-width ratio.
    """
    ga = _cast(collection)
    box = shapely.bounds(ga).squeeze()
    #(xmin, xmax), (ymin, ymax) = box[[0, 2]].T, box[[1, 3]].T
    (xmin, xmax), (ymin, ymax) = box[:, [0, 2]].T, box[:, [1, 3]].T
    widths, heights = np.abs(xmax - xmin), np.abs(ymax - ymin)
    return widths - heights

def boundary_amplitude(collection):
    """Calculate the boundary amplitude of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The boundary amplitude.
    """
    ga = _cast(collection)
    return shapely.measurement.length(
        shapely.convex_hull(ga)
    ) / shapely.measurement.length(ga)

def convex_hull_ratio(collection):
    """Calculate the convex hull ratio of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The convex hull ratio.
    """
    ga = _cast(collection)
    return shapely.area(ga) / shapely.area(shapely.convex_hull(ga)) 

def fractal_dimension(collection, support = "hex"):
    """Calculate the fractal dimension of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The fractal dimension.
    """
    ga = _cast(collection)
    P = shapely.measurement.length(ga)
    A = shapely.area(ga)
    if support == "hex":
        return 2 * np.log(P / 6) / np.log(A / (3 * np.sin(np.pi / 3)))
    elif support == "square":
        return 2 * np.log(P / 4) / np.log(A)
    elif support == "circle":
        return 2 * np.log(P / (2 * np.pi)) / np.log(A / np.pi)
    else:
        raise ValueError(
            "The support argument must be one of 'hex', 'circle', or 'square', "
            f"but {support} was provided."
        )

def squareness(collection):
    """Calculate the squareness of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The squareness.
    """
    ga = _cast(collection)
    return ((np.sqrt(shapely.area(ga)) * 4) / shapely.length(ga)) ** 2

def rectangularity(collection):
    """Calculate the rectangularity of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The rectangularity.
    """
    ga = _cast(collection)
    return shapely.area(ga) / shapely.area(shapely.minimum_rotated_rectangle(ga))

def shape_index(collection):
    """Calculate the shape index of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The shape index.
    """
    ga = _cast(collection)
    return np.sqrt(shapely.area(ga) / np.pi) / shapely.minimum_bounding_radius(ga)

def equivalent_rectangular_index(collection):
    """Calculate the equivalent rectangular index of a collection of polygons.
    
    Parameters
    ----------
    collection : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A collection of polygons.
    
    Returns
    -------
    float
        The equivalent rectangular index.
    """
    ga = _cast(collection)
    box = shapely.minimum_rotated_rectangle(ga)
    return np.sqrt(shapely.area(ga) / shapely.area(box)) * (
        shapely.length(box) / shapely.length(ga)
    )



