import statistics
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import math
import cv2
import scipy.signal
import scipy.ndimage
import scipy.io
import sklearn.preprocessing
from PIL import Image
import os

# function for adding 0 padding to matrix
def Add_Padding(Img_Matrix):
    for i in range(len(Img_Matrix)+1):
        # initialize 1D row
        row=[]
        # Add padding at start of matrix
        if i==0:
            # loop for column
            for j in range(len(Img_Matrix[i])):
                # Append 0 value in row
                row.append(0)
            # Insert row in original image matrix at start
            Img_Matrix.insert(0,row)
        # Add padding at end of matrix
        if i== len(Img_Matrix)-1:
            # Add padding at start and end of matrix in each row
            L=len(Img_Matrix[i])+2
            for j in range(L):
                # Append 0 value in row
                row.append(0)
            # Insert 0 in original image matrix at end of each row
            Img_Matrix.insert(i+1,row)
        # Insert 0 in original image matrix at start of each row    
        Img_Matrix[i].insert(0,0)
        # Append padding in original matrix row
        Img_Matrix[i].append(0)
    return Img_Matrix
  
# function for second derivative in vertical direction
def Second_Derivative_Vertical(Img_Matrix):
    # find number of rows
    rows = len(Img_Matrix)
    # find number of column
    col = len(Img_Matrix[0])-1
    Vertical_first_Image = []
    # loop for row
    for i in range(1,rows-1):
        # initialize 1d row
        first_D = []
        # loop for column
        for j in range(0,col-1):
            first = (Img_Matrix[i][j+2] + Img_Matrix[i][j] - 2*(Img_Matrix[i][j+1]))
            first_D.append(first)
        Vertical_first_Image.append(first_D)
    return Vertical_first_Image
  
# function for second derivative in horizontal direction
def Second_Derivative_Horizontal(Img_Matrix):
  # find number of rows
  rows = len(Img_Matrix)
  # find number of column
  col = len(Img_Matrix[0])
  Horizontal_second_Image = []
  # loop for row
  for i in range(0,rows-2):
      # initialize 1d row
      first_D = []
      # loop for column
      for j in range(1,col-1):
          first = (Img_Matrix[i][j] + Img_Matrix[i+2][j] - 2*(Img_Matrix[i+1][j]))
          first_D.append(first)
      Horizontal_second_Image.append(first_D)
  return Horizontal_second_Image

# function for second derivative in diagonal "\" direction
def Second_Derivative_Diagonal_TopLeft_BottomRight(Img_Matrix):
    # find number of rows
    rows = len(Img_Matrix)
    # find number of column
    col = len(Img_Matrix[0])
    diagonal_second_Image = []
    # loop for row
    for i in range(1, rows-1):
        # initialize 1d row
        first_D = []
        # loop for column
        for j in range(1, col - 1):
            first = (Img_Matrix[i - 1][j - 1] + Img_Matrix[i + 1][j + 1] - 2*(Img_Matrix[i][j])) / math.sqrt(2)
            first_D.append(first)
        diagonal_second_Image.append(first_D)
    return diagonal_second_Image
  
# function for second derivative in diagonal "/" direction
def Second_Derivative_Diagonal_TopRight_BottomLeft(Img_Matrix):
    # find number of rows
    rows = len(Img_Matrix)
    # find number of column
    col = len(Img_Matrix[0])
    diagonal_second_Image = []
    # loop for row
    for i in range(1, rows-1):
        # initialize 1d row
        first_D = []
        # loop for column
        for j in range(1, col - 1):
            first = (Img_Matrix[i - 1][j + 1] + Img_Matrix[i + 1][j - 1] - 2*(Img_Matrix[i][j])) / math.sqrt(2)
            first_D.append(first)
        diagonal_second_Image.append(first_D)
    return diagonal_second_Image
  
# function for normalizing array data to 0-255 range
def NormalizeForImageSaving(data):
    return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255.9).astype(np.uint8)

def psfmat(n, sigma):
    # create a point spread matrix with width sigma and length n
    u = [np.arange(-n, n+1)]
    v = np.exp((-(np.transpose(u)/sigma) ** 2) / 2)
    v = v / np.sum(v)
    S = np.zeros((n, n))
    nv = len(v)
    for i in range(1, n+1):
        u = n + 1
        l2 = max(n - i + 2, 1) - 1
        u2 = min(n + n - i + 1, nv)
        S[i-1][0:u] = v[l2:u2, 0]
    return S

def CG2(U, G1, G2, kappa, lmbda, delta):
    R = U
    P = R
    n1 = G1.shape[1]
    n2 = G2.shape[1]
    X = np.zeros((n1, n2))
    D1 = np.transpose(np.diff(np.eye(n1)))
    D2 = np.transpose(np.diff(np.eye(n2)))
    V1 = lmbda * np.transpose(D1) @ D1
    V2 = lmbda * np.transpose(D2) @ D2

    for it in range(1, 101):
        Q = G1 @ P @ G2 + kappa * P + V1 @ P + P @ np.transpose(V2)
        alpha = np.sum(R[:] ** 2) / np.sum(P[:] * Q[:])
        X = X + alpha * P
        Rnew = R - alpha * Q
        rs1 = np.sum(R[:] ** 2)
        rs2 = np.sum(Rnew[:] ** 2)
        beta = rs2 / rs1
        P = Rnew + beta * P
        R = Rnew
        rms = np.sqrt(rs1 / (n1 * n2))
        if rms < delta:
            break
    return X

# get images from input folder
def get_images():
  input_path = "./input"  
  files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
  imgs = []
  for file in files:
    valid, extension = is_valid_file(file)
    if valid:
      imgs.append((file, extension))
  return imgs

def is_valid_file(filename):
  name, extension = os.path.splitext(filename)
  valid_extensions = [".dat", ".mat", ".jpg", ".jpeg", ".png", ".bmp", ".gif"]
  return extension in valid_extensions, extension

def load_img(filename, extension):
    input_path = "./input"
    filepath = f"{input_path}/{filename}"
    filename = os.path.splitext(filename)[0] # remove extension from filename
    loaded_imgs = []
    match extension:
        case ".dat":
            imgs = np.loadtxt(filepath)
            shape = imgs.shape
            # CHANGE THE 64 BELOW AND ASK FOR USER INPUT IF NEEDED
            end_index = int(shape[0] / 64)
            for i in range(0, end_index):
                start_index = i * 64
                img = imgs[start_index:start_index+64, :]
                loaded_imgs.append(img)
        case ".mat":
            mat = scipy.io.loadmat(filepath)
            loaded_imgs.append(mat[filename])
        case _:
            loaded_imgs.append("none")
    return loaded_imgs

def enhance_image(filename, extension):
    img_data = load_img(filename, extension)
    enhanced_imgs = []
    
    for img in img_data:
        if (img != "none"):
            # denoising
            img = NormalizeForImageSaving(img)
            gaussianMatrix = [[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]]
            cv2denoise = cv2.fastNlMeansDenoising(NormalizeForImageSaving(img), None, 3)
            img = scipy.signal.convolve2d(cv2denoise, gaussianMatrix, boundary="fill")
            
            # second derivative
            SVI = np.array(Second_Derivative_Vertical(img))
            SHI = np.array(Second_Derivative_Horizontal(img))
            SDTLI = np.array(Second_Derivative_Diagonal_TopLeft_BottomRight(img))
            SDTRI = np.array(Second_Derivative_Diagonal_TopRight_BottomLeft(img))

            # combine 4 derivative images
            minArray = []
            for i in range(0, len(SVI)):
                newMinRow = []
                for j in range(0, len(SVI[i])):
                    minimum = min(SVI[i][j], SHI[i][j], SDTLI[i][j], SDTRI[i][j])
                    newMinRow.append(minimum)
                minArray.append(newMinRow)
            
            # normalize image
            arrayMin = min(min(minArray))
            normArray = minArray / (-1 * arrayMin)
            
            # threshold image, set all values above 0 to 0
            for i in range(0, len(normArray)):
                for j in range(0, len(normArray[i])):
                    element = normArray[i][j]
                    if element >= 0:
                        normArray[i][j] = 0
                        
            # get the 5% minimum value of the image
            flat = np.ndarray.flatten(normArray)
            sortedArray = np.sort(flat)
            (dim1, dim2) = normArray.shape # get dimensions of image
            total_pixels = dim1 * dim2 # get total number of pixels
            fivePercentMin = sortedArray[round((0.005 * total_pixels)) - 1] # get 5% minimum value

            # threshold image, set all values below 5% minimum to -1, then linear stretch the rest
            for i in range(0, len(normArray)):
                for j in range(0, len(normArray[i])):
                    element = normArray[i][j]
                    if element <= fivePercentMin:
                        normArray[i][j] = -1
                    else:
                        normArray[i][j] = element / (-1 * fivePercentMin)
        enhanced_imgs.append(NormalizeForImageSaving(normArray))
    return enhanced_imgs

def superresolution(img, upscale_factor):
    (n1, n2) = img.shape
    p1 = upscale_factor * n1
    p2 = upscale_factor * n2
    sigma = 0.053 * upscale_factor
    S1a = psfmat(p1, sigma)
    S2a = psfmat(p2, sigma)
    S1 = np.kron(np.eye(n1), np.ones((1, upscale_factor))) @ S1a
    S2 = np.kron(np.eye(n2), np.ones((1, upscale_factor))) @ S2a
    
    U = np.transpose(S1) @ img @ S2
    G1 = np.transpose(S1) @ S1
    G2 = np.transpose(S2) @ S2
    kappa = 0.01
    lmbda = 100
    delta = 0.001
    X = CG2(U, G1, G2, kappa, lmbda, delta)
    return X

def main():
    img_files = get_images()
    upscale_factor = 8
    output_num = 1
    for file in img_files:
        enhanced_imgs = enhance_image(file[0], file[1])
        for img in enhanced_imgs:
            enhanced_super_img = NormalizeForImageSaving(superresolution(img, upscale_factor))
            Image.fromarray(enhanced_super_img).save(f"./output/{output_num}.png")
            output_num += 1
    return

main()