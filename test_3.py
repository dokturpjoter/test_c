import matplotlib.pyplot as plt

x = range(0,20)
y = [i*i for i in x]

plt.plot(x,y)
plt.close()