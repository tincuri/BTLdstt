import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import sympy as sp
from PIL import Image, ImageOps
import os
#Process the Image into a flat 1D array
def ProcessImage(img):
    Grascale = ImageOps.grayscale(img)
    Facematrix = im.pil_to_array(Grascale)
    FlatFaceArray = np.concat(Facematrix,axis=None)
    return FlatFaceArray

#Make the Data Matrix

def Datamatrix(clsdir: str, num_files: int):
    for i in range(num_files):
        imgdir = clsdir +"\\" + str(i+1) + ".pgm"
    #Nhap hinh anh
        Face = Image.open(imgdir)
    #Process hinh anh
        FlatFaceArray = ProcessImage(Face)
        if i == 0:
            FaceMatrix = FlatFaceArray
        else:
            FaceMatrix = np.column_stack((FaceMatrix,FlatFaceArray))
    return FaceMatrix

#Make a Data Matrix for a class k
def clsMatrix(k: int):
    clsdir = "archive\\s" + str(k)
    num_files = sum(1 for entry in os.scandir(clsdir) if entry.is_file())
    return Datamatrix(clsdir, num_files)
clsMatrix(2).shape

#number of class
num_class = sum(1 for entry in os.scandir("archive") if not entry.is_file())

for i in range(3):
    if i == 0:
        FullMatrix = 4
        a = 0
    else:
        FullMatrix = 2
print(FullMatrix)