#MC920/MO443 - Introducao ao processamento digital de imagem
#Professor Helio Pedrini
#IC/UNICAMP - 1s2019

#Raysa Masson Benatti
#176483
#Trabalho 5: aplicar tecnicas de agrupamento de dados (aprendizado de maquina nao supervisionado)
#para reduzir (quantizar) o numero de cores de uma img colorida
#procurando manter a qualidade da aparencia geral da img de entrada 

import numpy as np 
import cv2 
import time 

img = cv2.imread('baboon.png')
Z = img.reshape((-1,3))

#converte para float32
Z = np.float32(Z)

#define criterios, numero de clusters (K) e aplica kmeans()
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 512
t_inicio = time.time()
ret, label, center=cv2.kmeans(Z, K, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
t_fim = time.time()
print "Tempo gasto com K = " + str(K) + ": ", t_fim - t_inicio

#converte de volta para uint8 e refaz imagem original
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imwrite("res_"+ str(K) +".png", res2)