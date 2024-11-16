import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import sympy as sp
from PIL import Image, ImageOps
import os
#Size of the image
Face = Image.open("archive\\s1\\1.pgm") 
FaceArray = im.pil_to_array(Face)
Height, Width = FaceArray.shape
#Number of classes
num_class = sum(1 for entry in os.scandir("archive") if not entry.is_file())
#Number of files in a class
num_files = sum(1 for entry in os.scandir("archive\\s1") if entry.is_file())-3
#Total number of files
num_total = num_files*num_class

#Process the Image into a flat 1D array
def ProcessImage(img):
    Grayscale = img.convert('L')
    Facematrix = im.pil_to_array(Grayscale)/255
    FlatFaceArray = np.concat(Facematrix,axis=None)

    return FlatFaceArray

#Make the Data Matrix
def Datamatrix(clsdir: str, num_files: int):
    for i in range(num_files):
        imgdir = clsdir +"\\" + str(i+1) + ".pgm"
    #Import image
        Face = Image.open(imgdir)
        FlatFaceArray = ProcessImage(Face)
    #Combine each 1D array in to a matrix
        if i == 0:
            FaceMatrix = FlatFaceArray
        else:
            FaceMatrix = np.column_stack((FaceMatrix,FlatFaceArray))
    return FaceMatrix

#Make a Data Matrix for a class k
def ClassMatrix(k: int):
    clsdir = "archive\\s" + str(k)
    return Datamatrix(clsdir, num_files)

#More efficent way to calculate the svd of the datamatrix
def svd(X):
    XtX = X.T @ X
    V, S, Vt = np.linalg.svd(XtX)
    S_matrix = np.sqrt(np.diag(S))
    S_inv = np.linalg.inv(S_matrix)
    U = X @ V @ S_inv
    return U, np.sqrt(S), Vt

#Make the whole data matrix
for i in range(num_class):
    if i == 0:
        FullMatrix = ClassMatrix(i+1)
    else:
        FullMatrix = np.append(FullMatrix, ClassMatrix(i+1),axis=1)

#The average face
MeanFace=np.mean(FullMatrix, axis=1)
ColumnMeanFace= MeanFace[:, np.newaxis]
print(MeanFace)

#Minus the mean
CenteredFullMatrix=FullMatrix  - ColumnMeanFace
#Find the Eigenface of the Data
U, S, Vt = svd(CenteredFullMatrix)
#Find the needed amount of eigenvalues
eigsum=np.sum(S)
csum=0
for i in range(S.shape[0]):
    csum +=S[i]
    if csum > 0.9*eigsum:
        e90=i
        break

#Reducing the eigenvalues and vector to the needed amount
U_reduced = U[:,:e90]
S_reduced= np.sqrt(np.diag(S[0:e90]))

#Projecting a vector in to the facespace
def Facespace(vector):
    return U_reduced.T @ vector

#the average of a class k in facespace
def avgClassOmega(k: int):
    S = 0
    for i in range(num_files):
        S+= Facespace(ClassMatrix(k)[:,i]-MeanFace)
    return S/len(range(num_files))

#Find the threshold for the picture to be indentified as a face in class k
def Classepsilon(k: int):
    Cepsilon = 0
    for i in range(num_files):
        e=np.linalg.norm(Facespace(ClassMatrix(k)[:,i] - MeanFace)-avgClassOmega(k))
        if e > Cepsilon:
            Cepsilon = e
    return Cepsilon*2

#Find the threshold for the picture to be identified as a face, let that be epsilon.
epsilon=0
#Array of a facevector after being transform in facespace
def Ffacearray(facearray):
    return Facespace(facearray) @ U_reduced.T
#Find the longest distance between the data image and its projection in facespace
for i in range(num_total):
    facearray = CenteredFullMatrix[:,i]
    e = np.linalg.norm(facearray-Ffacearray(facearray))
    if e > epsilon:
        epsilon = e
#Let the threshold be larger than epsilon to catch the edge case
epsilon = 4*epsilon

#Let the image that need to be specify is input
def Facedetect(imgdir: str ):
    image=Image.open(imgdir)
    facearray = ProcessImage(image)-MeanFace
    #Find the distant between the input image and its projection in facespace
    dist = np.linalg.norm(facearray - Ffacearray(facearray))
    if epsilon > dist:
        print("this image contains a face")
    else: 
        print("this image does not contain a face")
    print(dist)
#Classify the image into the known class
def EigenClassify(imgdir: str):
    image = Image.open(imgdir)
    facearray = ProcessImage(image) - MeanFace
    dist = 1000
    for i in range(num_class):
        dist2 = np.linalg.norm(Facespace(facearray) - avgClassOmega(i+1))
        if dist2 < dist:
            dist = dist2
            k = i
    if dist <= Classepsilon(k+1)*2:
        print(f"This image is in class {k+1}")
    else:
        print("This image is not identified")

#Fisher face

#SVD of the datamatrix
U, S, V = svd(CenteredFullMatrix)

#Find the dimetion needed for LDA
W_pca = U[:,:num_total - num_class]

#Mean of a class
def MeanClassFace(k: int):
    MeanFace=np.mean(ClassMatrix(k), axis=1)
    return MeanFace[:, np.newaxis]
#Find Sb
for i in range(num_class):
    if i == 0:
        Xb = MeanClassFace(i+1) - ColumnMeanFace
    else:
        Xb = np.column_stack((Xb,MeanClassFace(i+1)-ColumnMeanFace))
Xb2 = W_pca.T @ Xb
Sb = num_files * Xb2 @ Xb2.T
#Find Sw
for i in range(num_class):
    if i == 0:
        Xw = ClassMatrix(i+1) - MeanClassFace(i+1)
        Xw2 = W_pca.T @ Xw
        Sw = Xw2 @ Xw2.T
    else:
        Xw = ClassMatrix(i+1) - MeanClassFace(i+1)
        Xw2 = W_pca.T @ Xw
        Sw += Xw2 @ Xw2.T

#LDA
U, S, Vt = np.linalg.svd(np.linalg.inv(Sw) @ Sb)
#Reduce the matrix dimention to the rank of Sb
dim = np.linalg.matrix_rank(Sb)
W_lda= U[:, :dim] 
S_lda= np.diag(S[0:dim])
#Find W_opt
W_opt=W_pca @ W_lda
#Find the projection of a facevector to fisher face space
def Fisherfacespace(vec):
    return W_opt.T @ vec
#the average of a class k in facespace
def avgFisherClass(k: int):
    S = 0
    for i in range(num_files):
        S+= Fisherfacespace(ClassMatrix(k)[:,i]-MeanFace)
    return S/len(range(num_files))

#Find the threshold for the picture to be indentified as a face in class k
def FisherClassepsilon(k: int):
    Cepsilon = 0
    for i in range(num_files):
        e=np.linalg.norm(Fisherfacespace(ClassMatrix(k)[:,i]-MeanFace)-avgFisherClass(k))
        if e > Cepsilon:
            Cepsilon = e
    return Cepsilon*2
#Classify the input to the known class
def FisherClassify(imgdir: str):
    image = Image.open(imgdir)
    facearray = ProcessImage(image) - MeanFace
    dist = 1000
    for i in range(num_class):
        dist2 = np.linalg.norm(Fisherfacespace(facearray) - avgFisherClass(i+1))
        if dist2 < dist:
            dist = dist2
            k = i
    if dist <= FisherClassepsilon(k+1)*10:
        print(f"This image is in class {k+1}")
    else:
        print("This image is not identified")
        print(dist2)


