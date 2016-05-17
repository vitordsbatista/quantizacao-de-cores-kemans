#Algoritmo que seleciona as cores dominantes atraves da clusterizacao utilizando o algoritmo k-means
from matplotlib import pyplot as plt
import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import cv2
import time

#Importa a imagem
#img = misc.imread("itest.png")
img = misc.imread("test.png")
x, y, z = img.shape
#img = img/255
#Cluesterizacao das cores (quantizacao) com o K-Mean

#Muda a estrutura da imagem, de tridimensional (linha, coluna e cor), para bidimensional (linha * coluna, cor)
img1= img.reshape(x * y, 3)
#Numero de clusters
nclusters = 5 
#Kmeans

kmeans = KMeans(n_clusters=nclusters)

t0 = time.time()
kmeans.fit(img1)
print "tempo do fit", time.time() - t0

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print centroids
print labels

#Criacao das cores predominantes da imagem (retirei de um site, link na pagina inicial)
#Histograma
nlabels = np.arange(len(np.unique(labels)) + 1)
(hist, _) = np.histogram(labels, bins=nlabels)
hist = hist.astype("float")
hist /= hist.sum()

#Barras com a imagem
mc = np.zeros([50, 300, 3])
mcY = mc.shape[1]
for i in range(nclusters):
    mc[:,i*mcY/nclusters:(i+1)*mcY/nclusters] = centroids[i].astype("uint8")

#Imagem quantizada
#Imagem quantizada
img2 = centroids[labels].reshape(x, y, 3).astype("uint8")
#print bar
#print mc

#Plot das imagens
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(img)
ax1.set_title("Imagem original")

ax4 = fig.add_subplot(122)
ax4.imshow(img2)
ax4.set_title("Custerizacao")

plt.show()
