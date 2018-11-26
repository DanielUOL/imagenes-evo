import random
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import imageio

class Individual:

	def __init__(self,n,w,h):
		self.n = n
		self.w = w
		self.h = h
		adn = []
		for i in range(n):
			p1 = (random.randint(0,w),random.randint(0,h))
			p2 = (random.randint(p1[0]-30,p1[0]+30),random.randint(p1[1]-30,p1+30))
			p3 = (random.randint(p1[0]-30,p1[0]+30),random.randint(p1[1]-30,p1+30))
			color = random.randint(0,255)
			adn.append([p1,p2,p3,color])


#imagen = Individual(100,480,360)

#face = misc.face()
#misc.imsave('face.png',face)
im = imageio.imread('face.png')
print(im)
plt.imshow(im)
plt.show()


