#!/usr/bin/env python3

import os
import sys
import numpy as np


class Detection:
    """
    Detection object handles all the properties of a detected object    
    """

    def __init__(self, result, width, height):
        ### Initialize any class variables desired ###
        self.new = None
        
        #confidece
        self.conf = np.transpose(result[0])[2]
        
        self.xmin = int(np.transpose(result[0])[3][0] * width)
        self.ymin = int(np.transpose(result[0])[4][0] * height)

        self.xmax = int(np.transpose(result[0])[5][0] * width)
        self.ymax = int(np.transpose(result[0])[6][0] * height)

        return