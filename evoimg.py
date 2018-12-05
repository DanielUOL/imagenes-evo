import random
import numpy as np
import cv2
import copy
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
		for i in range(self.n//8):
			point = random.randint(0,self.n-2)
			center = (random.randint(0,self.w),random.randint(0,self.h))
			radius = random.randint(0,self.r)
			color = 255
			self.adn[point] = [center,radius,(color,color,color)]

	def crossover(self,I2):
		point = random.randint(1,self.n-2)
		adn1 = []
		adn2 = []
		selfadn = copy.deepcopy(self.adn)
		I2adn = copy.deepcopy(I2.adn)
		for i in range(0,point):
			adn1.append(selfadn[i])
			adn2.append(I2adn[i])
		
		for i in range(point,self.n):
			adn1.append(I2adn[i])
			adn2.append(selfadn[i])
		

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
		img[img==255] = 1
		#err = np.sum((opt.astype("float") - img.astype("float")) ** 2)
		#err /= float(opt.shape[0] * opt.shape[1]) 
		mres = opt^img
		err = mres.sum()
		self.fit = err


def toImage(imgM,name):
	img = np.zeros((583,480,3), np.uint8)
	for circle in imgM.adn:
		cv2.circle(img,circle[0],circle[1],circle[2],-1)
	cv2.imwrite(name,img)


optimal = cv2.imread('manzana.png')
optimal = cv2.cvtColor(optimal, cv2.COLOR_BGR2GRAY)
binaryopt = np.copy(optimal)
binaryopt[optimal>=200] = 1
binaryopt[optimal<200] = 0

C = 100  # Cantidad de circulos
R = 52  # Radio
N = 20  # Individuos
pm = 0.8  # Probabilidad de mutacion
width = 480  # Ancho
height = 583  # Alto
G = 5000 # Numero de generaciones
n = 1

Population = [Individual(C,width,height,R) for i in range(N)]
for ind in Population:
	ind.fitness(binaryopt)

orig = Population[0]
nombre = "originalC100R52I20G5000F"+str(Population[0].fit)+".png"
toImage(orig,nombre)


for i in range(G):
	print("Generacion ",i)
	offspring = []
	plebada = []
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

		H1.fitness(binaryopt)
		H2.fitness(binaryopt)
		offspring.append(H1)
		offspring.append(H2)

	plebada = Population + offspring
	plebada.sort(key=lambda x: x.fit)
	Population = plebada[:N]
	print(Population[0].fit)
	name = str(n)+".png"
	#toImage(Population[0],name)
	n+=1

nombre = "C100R52I20G5000"+str(Population[0].fit)+".png"
toImage(Population[0],nombre)
