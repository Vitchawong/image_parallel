# runtime of parallel
import matplotlib.pyplot as plt

%matplotlib inline



a = [200, 300, 500, 800]
b = [1.3155276775360107,1.529815673828125,1.8073720932006836, 2.2616989612579346]
plt.xlabel('Image Size')
plt.ylabel('Runtime')

plt.plot(a,b)

# sequencial
c = [200, 300, 500, 800]
d = [0.24578857421875,0.5436396598815918,1.544672966003418,3.963372230529785]
plt.xlabel('Image Size')
plt.ylabel('Runtime')

plt.plot(c,d)
