# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:21:28 2020

@author: Srikanth Guptha
"""
#
#  // 3D model points.
#std::vector<cv::Point3d> model_points;
#model_points.push_back(cv::Point3d(f, f, f));  

from mpl_toolkits import mplot3d           
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def _get_full_model_points(filename):
    
    """Get all 68 3D model points from file"""
    raw_value = []
    with open(filename) as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T

    # Transform the model into a front view.
    model_points[:, 2] *= -1

    return model_points


filename= 'assets/model32.txt'
points =  _get_full_model_points(filename)

# Data for a three-dimensional line

fig = plt.figure()

ax = plt.axes(projection='3d')

zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)


ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points

zdata = ([row[0] for row in points])
xdata = ([row[1] for row in points])
ydata = ([row[2] for row in points])

ax.scatter3D(xdata[:], ydata[:], zdata[:], cmap='red');
