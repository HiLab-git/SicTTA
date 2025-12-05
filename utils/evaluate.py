# -*- coding: utf-8 -*-
"""
Evaluation module for segmenation tasks.
"""
from __future__ import absolute_import, print_function
from cmath import nan
import csv
import os
import sys
import math
import pandas as pd
import random
# import GeodisTK
import configparser
import numpy as np
from scipy import ndimage
# from pymic.io.image_read_write import *
# from pymic.util.image_process import *
# from pymic.util.parse_config import parse_config


def binary_dice(s, g, resize = False):
    """
    Calculate the Dice score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    :param resize: (optional, bool) 
        If s and g have different shapes, resize s to match g.
        Default is `True`.
    
    :return: The Dice value.
    """
    assert(len(s.shape)== len(g.shape))
    if(resize):
        size_match = True
        for i in range(len(s.shape)):
            if(s.shape[i] != g.shape[i]):
                size_match = False
                break
        # if(size_match is False):
        #     s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-5)/(s1 + s2 + 1e-5)
    return dice

def binary_iou(s,g):
    """
    Calculate the IoU score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    
    :return: The IoU value.
    """
    assert(len(s.shape)== len(g.shape))
    intersecion = np.multiply(s, g)
    union = np.asarray(s + g >0, np.float32)
    iou = (intersecion.sum() + 1e-5)/(union.sum() + 1e-5)
    return iou

# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    Get edge points of a binary segmentation result.

    :param img: (numpy.array) a 2D or 3D array of binary segmentation.
    :return: an edge map. 
    """
    dim = len(img.shape)
    if(dim == 2):
        strt = ndimage.generate_binary_structure(2,1)
    else:
        strt = ndimage.generate_binary_structure(3,1)
    ero  = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8) 
    return edge 


def binary_hd95(s, g, spacing = None):
    """
    Get the 95 percentile of hausdorff distance between a binary segmentation 
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    
    :return: The HD95 value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    img = np.zeros_like(s)
    if(image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif(image_dim ==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1)*0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2)*0.95)]
    return max(dist1, dist2)


def binary_assd(s, g, spacing = None):
    """
    Get the Average Symetric Surface Distance (ASSD) between a binary segmentation 
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    
    :return: The ASSD value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    img = np.zeros_like(s)
    if(image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif(image_dim ==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum() ) / (ns + ng + 1e-7) 
    
    if assd > 10:
        assd = 10
    return assd

# relative volume error evaluation
def binary_relative_volume_error(s, g):
    """
    Get the Relative Volume Error (RVE) between a binary segmentation 
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.

    :return: The RVE value.
    """
    s_v = float(s.sum())
    g_v = float(g.sum())
    assert(g_v > 0)
    rve = abs(s_v - g_v)/g_v
    return rve

def get_binary_evaluation_score(s_volume, g_volume, metric):
    """
    Evaluate the performance of binary segmentation using a specified metric. 
    The metric options are {`dice`, `iou`, `assd`, `hd95`, `rve`, `volume`}. 

    :param s_volume: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g_volume: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.
    :param metric: (str) The metric name. 

    :return: The metric value.
    """
    if(len(s_volume.shape) == 4):
        assert(s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if(s_volume.shape[0] == 1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()
    if(metric_lower == "dice"):
        score = binary_dice(s_volume, g_volume)
    elif(metric_lower == "iou"):
        score = binary_iou(s_volume,g_volume)
    elif(metric_lower == 'assd'):
        score = binary_assd(s_volume, g_volume)
    elif(metric_lower == "hd95"):
        score = binary_hd95(s_volume, g_volume)
    elif(metric_lower == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score

def get_multi_class_evaluation_score(s_volume, g_volume, label_list, metric):
    score_list = []
    for label in range(1,label_list):
        temp_score = get_binary_evaluation_score(s_volume == label, g_volume == label, metric)
        score_list.append(temp_score)
    return score_list

