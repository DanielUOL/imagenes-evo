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
			color = 255
			self.adn.append([center,radius,(color,color,color)])

	def mutation(self):
		for i in range(self.n//2):
			point = random.randint(0,self.n-2)
			center = (random.randint(0,self.w),random.randint(0,self.h))
			radius = random.randint(0,self.r)
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


	def fitness(self,opt):
		value = 0
		img = np.zeros((self.h,self.w,3), np.uint8)

		for circle in self.adn:
			cv2.circle(img,circle[0],circle[1],circle[2],-1)

		for i in range(self.h):
			for j in range(self.w):
				if img[i][j][0] != opt[i][j]:
					value +=1
		return value

optimal = cv2.imread('manzana.png',0)

C = 600  # Cantidad de circulos
R = 20  # Radio
N = 100  # Individuos
pm = 0.8  # Probabilidad de mutacion
width = 480  # Ancho
height = 583  # Alto
G = 100 # Numero de generaciones

Population = [Individual(C,width,height,R) for i in range(N)]

for i in range(G):
	print("Generacion ",i)
	offspring = []
	for j in range(N//2):
		torneo = []
		for i in range(2):
			ind1 = Population[random.randint(0,N-1)]
			ind2 = Population[random.randint(0,N-1)]
			if ind1.fitness(optimal) < ind2.fitness(optimal):
				torneo.append(ind1)
			else:
				torneo.append(ind2)

		H1,H2 = torneo[0].crossover(torneo[1])

		if random.random() <= pm:
			H1.mutation()
		if random.random() <= pm:
			H2.mutation()

		offspring.append(H1)
		offspring.append(H2)

	plebada = Population + offspring
	plebada.sort(key=lambda x: -x.fitness(optimal))
	population = plebada[:N]



img1 = np.zeros((583,480,3), np.uint8)
for circle in Population[0].adn:
	cv2.circle(img1,circle[0],circle[1],circle[2],-1)
cv2.imwrite('original.png',img1)

"""
img1 = np.zeros((583,480,3), np.uint8)
for circle in imagen.adn:
	cv2.circle(img1,circle[0],circle[1],circle[2],-1)
cv2.imwrite('original.png',img1)
"""