#MC920/MO443 - Introducao ao processamento digital de imagem
#Professor Helio Pedrini
#IC/UNICAMP - 1s2019

#Raysa Masson Benatti
#176483
#Trabalho 4: aplicar tecnicas de deteccao de pontos de interesse
#para registrar um par de imagens e criar uma imagem panoramica
#formada pela ligacao entre as imagens apos sua correspondencia 

import numpy as np 
import cv2 
import time 

#OBS - exige instalacao a parte de opencv-contrib-python
def sift(img1, img2, gray1, gray2, lim):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(gray1, None)
	kp2, des2 = sift.detectAndCompute(gray2, None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	good = [] 
	for m in matches:
		if m[0].distance < lim*m[1].distance:
			good.append(m)
	matches = np.asarray(good)

	lines = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv2.imwrite("sift_matches.jpg", lines)

	if len(matches[:,0]) >= 4:
		src = np.float32([kp1[m.queryIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
		dst = np.float32([kp2[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
		h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
	else:
		print("Nao ha correspondencias o suficiente!") 
		return 

	dst = cv2.warpPerspective(img1, h, (img2.shape[1] + img1.shape[1], img2.shape[0]))
	dst[0:img2.shape[0], 0:img2.shape[1]] = img2
	return dst, h  

def orb(img1, img2, gray1, gray2):

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
 
    lines = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv2.imwrite("orb_matches.jpg", lines)

    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
    	pts1[i,:] = kp1[match.queryIdx].pt
    	pts2[i,:] = kp2[match.trainIdx].pt

    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    dst = cv2.warpPerspective(img1, h, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    return dst, h

#--------------------------------------------------#
#Leitura das imagens e conversao para escala de cinza 
imgA = cv2.imread("foto2A.jpg", cv2.IMREAD_COLOR)
imgB = cv2.imread("foto2B.jpg", cv2.IMREAD_COLOR)
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)


print("Escolha o detector de pontos de interesse e descritores a ser utilizado.")
print("Digite: A para SIFT; B para ORB")
detector = raw_input()

if(detector == 'A'):
	lim = input("Escolha o limiar para selecao das melhores correspondencias (valor entre 0.01 e 0.5): ")
	t_inicio = time.time()
	res, h = sift(imgA, imgB, grayA, grayB, lim)
	cv2.imwrite("res_sift.jpg", res)
	t_fim = time.time()
	t = t_fim - t_inicio
	print "Tempo total de execucao (sift) =", t

if(detector == 'B'):
	t_inicio = time.time()
	res, h = orb(imgA, imgB, grayA, grayB)
	cv2.imwrite("res_orb.jpg", res)
	t_fim = time.time()
	t = t_fim - t_inicio
	print "Tempo total de execucao (orb) =", t

print(h)
