# Imports :
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# dowloanding the dataset :

mnist = fetch_openml('mnist_784', as_frame=False)
print(mnist)

print(mnist.data)
print(mnist.target)
len(mnist.data)
help(len)
print(mnist.data.shape)
print(mnist.target.shape)
mnist.data[0]
mnist.data[0][1]
mnist.data[:,1]
mnist.data[:100]

# ----------------------Trying diffrent methods & commandes (Visualization) ---------------------
print("----------------------Visualization :-------------------------------")

images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()