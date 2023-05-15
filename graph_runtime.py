# runtime of parallel
import matplotlib.pyplot as plt

%matplotlib inline



a = [200, 300, 500, 800]
b = [1.3155276775360107,1.529815673828125,1.8073720932006836, 2.2616989612579346]
plt.xlabel('Image Size')
plt.ylabel('Runtime')

plt.plot(a,b)
