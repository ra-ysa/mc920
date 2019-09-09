#MC920/MO443 - Introducao ao processamento digital de imagem
#Professor Helio Pedrini
#IC/UNICAMP - 1s2019

#Raysa Masson Benatti
#176483
#Trabalho 2: alterar os niveis de cinza de uma imagem por meio das tecnicas de
#pontilhado (halftoning) ordenado e pontilhado com difusao de erro

from scipy import misc
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import math 

#--------------------------------------------------#

#adiciona dois valores 
#de modo que o resultado nao seja menor que 0 nem maior que 255
def add(a, b):
	if (a+b > 255):
		return 255
	elif(a+b < 0):
		return 0
	else:
		return a + b 

#normaliza um valor [0,255] para um valor [0,9]
def norm3(a):
	return a * 9/255 

#normaliza um valor [0,255] para um valor [0,15]
def norm4(a):
	return a * 15/255

#mascara para pontilhado ordenado 3x3
mask3 = [[6, 8, 4], [1, 0, 3], [5, 2, 7]]

#mascara para pontilhado ordenado 4x4 (Bayer)
mask4 = [[0, 12, 3, 15], [8, 4, 11, 7],
		[2, 14, 1, 13], [10, 6, 9, 5]]

#--------------------------------------------------#

#produz uma imagem criada com a tecnica de meios-tons
#em pontilhado ordenado 3x3
def halftoning_3(img):
	N_img = img.shape[0] #numero de linhas = altura
	M_img = img.shape[1] #numero de colunas = largura

	img_res = img 
	
	#normaliza valores dos pixels
	#e os substitui
	#por branco se p(img) >= p(mask)
	#por preto se p(img) < p(mask)
	for i in range(N_img):
		for j in range(M_img):
			img_res[i][j] = norm3(img_res[i][j])
			if(img_res[i][j] >= mask3[i%3][j%3]):
				img_res[i][j] = 255
			else:
				img_res[i][j] = 0
	return img_res

#produz uma imagem criada com a tecnica de meios-tons
#em pontilhado ordenado de Bayer (4x4)
def halftoning_4(img):
	N_img = img.shape[0] #numero de linhas = altura
	M_img = img.shape[1] #numero de colunas = largura

	img_res = img 

	#normaliza valores dos pixels
	#e os substitui
	#por branco se p(img) >= p(mask)
	#por preto se p(img) < p(mask)
	for i in range(N_img):
		for j in range(M_img):
			img_res[i][j] = norm4(img_res[i][j])
			if(img_res[i][j] >= mask4[i%4][j%4]):
				img_res[i][j] = 255
			else:
				img_res[i][j] = 0
	return img_res
	
#--------------------------------------------------#

#pontilhado por difusao de erro de floyd-steinberg

def varredura(img, i, j, N_img, M_img):
	if(img[i][j] > 128):
		erro = img[i][j] - 255
		img[i][j] = 255
	else:
		erro = img[i][j] - 0
		img[i][j] = 0

	if(i != N_img-1 and j != 0 and j != M_img-1):
		img[i][j+1] = add(img[i][j+1],(7*erro/16))
		img[i+1][j+1] = add(img[i+1][j+1], (erro/16))
		img[i+1][j] = add(img[i+1][j], (5*erro/16))
		img[i+1][j-1] = add(img[i+1][j-1], (3*erro/16))

#varredura da esquerda para direita
def floydsteinberg_A(img):
	N_img = img.shape[0] #numero de linhas = altura
	M_img = img.shape[1] #numero de colunas = largura

	img_res = img 
	
	for i in range(N_img):
		for j in range(M_img):
			varredura(img_res, i, j, N_img, M_img)
	return img_res

#varredura alternada
def floydsteinberg_B(img):
	N_img = img.shape[0] #numero de linhas = altura
	M_img = img.shape[1] #numero de colunas = largura

	img_res = img 

	for i in range(N_img):
		if(i%2 == 0): #se linha par, varre da esq para dir
			for j in range(M_img):
				varredura(img_res, i, j, N_img, M_img)
		if(i%2 == 1): #se linha impar, varre da dir para esq 
			for j in range(M_img-1, -1, -1):
				varredura(img_res, i, j, N_img, M_img)
	return img_res 

#--------------------------------------------------#

#Leitura da imagem
img1 = cv2.imread("retina.pgm", 0) #le imagem em modo grayscale
img2 = cv2.imread("fiducial.pgm", 0)
img3 = cv2.imread("sonnet.pgm", 0)
img4 = cv2.imread("peppers.pgm", 0)

#--------------------------------------------------#
#Uso das funcoes desejadas

res1 = halftoning_3(img1)
cv2.imwrite("retina_ht_3.pbm", res1)

res2 = halftoning_4(img2)
cv2.imwrite("fiducial_ht_4.pbm", res2)

res3 = floydsteinberg_A(img3)
cv2.imwrite("sonnet_fsA.pbm", res3)

res4 = floydsteinberg_B(img4)
cv2.imwrite("peppers_fsB.pbm", res4)

#Display
"""
cv2.imshow("Imagem", res1)
cv2.waitKey(0)
"""
