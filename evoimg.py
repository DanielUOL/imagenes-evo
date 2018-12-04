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
		self.fit = 0
		self.adn = []
		for i in range(n):
			center = (random.randint(0,w),random.randint(0,h))
			radius = random.randint(r//4,r)
			color = 255
			self.adn.append([center,radius,(color,color,color)])

	def mutation(self):
		for i in range(self.n//6):
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

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#err = np.sum((opt.astype("float") - img.astype("float")) ** 2)
		#err /= float(opt.shape[0] * opt.shape[1]) 
		mres = opt^img
		err = mres.sum()/255
		self.fit = err

optimal = cv2.imread('manzana.png')
optimal = cv2.cvtColor(optimal, cv2.COLOR_BGR2GRAY)

C = 100  # Cantidad de circulos
R = 50  # Radio
N = 30  # Individuos
pm = 0.8  # Probabilidad de mutacion
width = 480  # Ancho
height = 583  # Alto
G = 1000 # Numero de generaciones

Population = [Individual(C,width,height,R) for i in range(N)]
for ind in Population:
	ind.fitness(optimal)

orig = Population[0].fit

for i in range(G):
	print("Generacion ",i)
	offspring = []
	for j in range(N//2):
		torneo = []
		for i in range(2):
			ind1 = Population[random.randint(0,N-1)]
			ind2 = Population[random.randint(0,N-1)]
			if ind1.fit < ind2.fit:
				torneo.append(ind1)
			else:
				torneo.append(ind2)

		H1,H2 = torneo[0].crossover(torneo[1])

		if random.random() <= pm:
			H1.mutation()
		if random.random() <= pm:
			H2.mutation()

		H1.fitness(optimal)
		H2.fitness(optimal)
		offspring.append(H1)
		offspring.append(H2)

	plebada = Population + offspring
	plebada.sort(key=lambda x: x.fit)
	print(plebada[0].fit)
	population = plebada[:N]

print
img1 = np.zeros((583,480,3), np.uint8)
for circle in Population[0].adn:
	cv2.circle(img1,circle[0],circle[1],circle[2],-1)
cv2.imwrite('original.png',img1)

print("antes",orig,"despues",Population[0].fit)
"""
img1 = np.zeros((583,480,3), np.uint8)
for circle in imagen.adn:
	cv2.circle(img1,circle[0],circle[1],circle[2],-1)
cv2.imwrite('original.png',img1)
"""