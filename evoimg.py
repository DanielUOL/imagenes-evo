import random
import numpy as np
import cv2
#from scipy import misc
#import matplotlib.pyplot as plt
#import imageio

class Individual:

	def __init__(self,n,w,h,r):
		self.n = n
		self.w = w
		self.h = h
		self.r = r
		self.adn = []
		for i in range(n):
			center = (random.randint(0,w),random.randint(0,h))
			radius = random.randint(0,r)
			#color = random.randint(0,255)
			color = 255
			self.adn.append([center,radius,(color,color,color)])

	def mutation(self):
		for i in range(self.n//2):
			point = random.randint(0,self.n-1)
			center = (random.randint(0,self.w),random.randint(0,self.h))
			radius = random.randint(0,self.r)
			#color = random.randint(0,255)
			color = 255
			self.adn[point] = [center,radius,(color,color,color)]

	def crossover(self,I2):
		point = random.randint(1,self.n-2)
		adn1 = []
		adn2 = []
		for i in range(0,point):
			adn1.append(self.adn[i])
			adn2.append(I2.adn[i])
		for i in range(point,self.n-1):
			adn1.append(I2.adn[i])
			adn2.append(self.adn[i])

		H1 = Individual(self.n,self.w,self.h,self.r)
		H2 = Individual(self.n,self.w,self.h,self.r)
		H1.adn = adn1
		H2.adn = adn2

		return H1,H2


def fitness(img,opt,h,w):
	value = 0
	for i in range(h):
		for j in range(w):
			#if img[i][j][0] != opt[i][j][0] or img[i][j][1] != opt[i][j][1] or img[i][j][2] != opt[i][j][2]:
			#	value +=1
			if img[i][j][0] != opt[i][j]:
				value +=1
	return value

optimal = cv2.imread('manzana.png',0)

n = 600
resolution = 20

imagen = Individual(n,480,583,resolution)
imagen2 = Individual(n,480,583,resolution)

#Imagen de padre 1
img1 = np.zeros((583,480,3), np.uint8)
for circle in imagen.adn:
	cv2.circle(img1,circle[0],circle[1],circle[2],-1)
cv2.imwrite('original.png',img1)

#dif = fitness(img1,optimal,480,583)
print(optimal[90][164])
print(optimal[230][320])
print(img1[0][0])

print("dif ",fitness(img1,optimal,583,480))
"""
#Imagen de padre 1 mutado
imagen.mutation()
img = np.zeros((583,480,3), np.uint8)
for circle in imagen.adn:
	cv2.circle(img,circle[0],circle[1],circle[2],-1)
cv2.imwrite('mutado.png',img)

#Imagen de padre 2
img = np.zeros((583,480,3), np.uint8)
for circle in imagen2.adn:
	cv2.circle(img,circle[0],circle[1],circle[2],-1)
cv2.imwrite('padre2.png',img)

H1,H2 = imagen.crossover(imagen2)

#Imagen de hijo 2
img = np.zeros((583,480,3), np.uint8)
for circle in H1.adn:
	cv2.circle(img,circle[0],circle[1],circle[2],-1)
cv2.imwrite('hijo1.png',img)

#Imagen de hijo 2
img = np.zeros((583,480,3), np.uint8)
for circle in imagen2.adn:
	cv2.circle(img,circle[0],circle[1],circle[2],-1)
cv2.imwrite('hijo2.png',img)

"""
