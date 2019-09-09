#MC920/MO443 - Introducao ao processamento digital de imagem
#Professor Helio Pedrini
#IC/UNICAMP - 1s2019

#Raysa Masson Benatti
#176483
#Trabalho 1: implementar filtros de imagens no dominio espacial e de frequencias

from scipy import misc
from scipy import ndimage
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import math 

#Filtro h5 (combinacao de h3 e h4)
def h5(img1, img2):
	N_img = img1.shape[0]
	M_img = img1.shape[1]
	res = np.zeros((M_img, N_img))

	for i in range(M_img):
	    for j in range(N_img):
        	res[i][j] = math.sqrt((img1[i][j] ** 2) + (img2[i][j] ** 2))
	return res

#--------------------------------------------------#
#Definicao das mascaras

#h1
mask1 = np.zeros((5, 5))
for i in range(5):
	for j in range(5):
		if((j==2 and (i==0 or i==4)) or
		  ((j==1 or j==3) and (i==1 or i==3)) or
		  ((j==0 or j==4) and (i==2))):
			mask1[i][j] = -1
		elif((j==2 and (i==1 or i==3)) or
		  ((j==1 or j==3) and (i==2))):
		    mask1[i][j] = -2	
mask1[2][2] = 16

#h2
mask2 = np.ones((5, 5))
for i in range(5):
	for j in range(5):
		if((j==2 and (i==0 or i==4)) or
		  (i==2 and(j==0 or j==4))):
			mask2[i][j] = 6/256
		elif(((i==0 or i==4) and (j==1 or j==3)) or  
		    ((i==1 or i==3) and (j==0 or j==4))):
			mask2[i][j] = 4/256
		elif((i==1 or i==3) and (j==1 or j==3)):
			mask2[i][j] = 16/256
		elif((j==2 and (i==1 or i==3)) or 
			(i==2 and (j==1 or j==3))):
			mask2[i][j] = 24/256
mask2[2][2] = 36/256

#h3
mask3 = np.zeros((3, 3))
for i in range(3):
	mask3[i][0] = -1
mask3[1][0] = -2
for i in range(3):
	mask3[i][2] = -mask3[i][0]

#h4
mask4 = mask3.transpose()

#--------------------------------------------------#

#Leitura da imagem
img1 = misc.imread("seagull.png") #deve incluir tambem True nos parametros para h2, h3 e h4
"""
img2 = misc.imread("butterfly.png", True)
img3 = misc.imread("city.png", True)
img4 = misc.imread("house.png", True)
img5 = misc.imread("baboon.png", True)
"""

#--------------------------------------------------#
#Uso das funcoes desejadas

#Filtro h1 
img_h1 = cv2.filter2D(img1, -1, mask1)
plt.imshow(img_h1, cmap='gray')
plt.show()

#Filtro h2
#Setar "True" na leitura de imagem ao utiliza-lo 
img_h2 = cv2.filter2D(img1, -1, mask2)
plt.imshow(img_h2, cmap='gray')
plt.show() 

#Filtro h3
#Setar "True" na leitura de imagem ao utiliza-lo 
img_h3 = cv2.filter2D(img1, -1, mask3)
plt.imshow(img_h3, cmap='gray')
plt.show() 

#Filtro h4
#Setar "True" na leitura de imagem ao utiliza-lo 
img_h4 = cv2.filter2D(img1, -1, mask4)
plt.imshow(img_h4, cmap='gray')
plt.show() 

#Filtro h5
img_h5 = h5(img_h3, img_h4)
plt.imshow(img_h5, cmap='gray')
plt.show() 

#Filtro gaussiano 
imgft = np.fft.fft2(img1) #calcula transformada de fourier da imagem 
imgft = np.fft.fftshift(imgft) #translada componente de frequencia zero para o centro do espectro

sigma = 30 #pode ser ajustado. quanto menor, maior a suavizacao
gauss = cv2.getGaussianKernel(img1.shape[0], sigma) #filtro gaussiano
gauss = gauss * np.transpose(gauss)
img_aux = imgft * gauss

img_res = np.fft.ifftshift(img_aux) #recuperacao da imagem
img_res = np.fft.ifft2(img_res)
img_res = np.abs(img_res)

#Display
plt.imshow(img_res, cmap='gray')
plt.show() 
