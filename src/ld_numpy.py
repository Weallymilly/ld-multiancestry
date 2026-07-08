import numpy as np
import math

G = np.array([[0, 1, 2],
              [0, 0, 1],
              [1, 1, 2],
              [2, 0, 0],
              [1, 2, 1]])

#Variants as rows, samples as columns

print(G.shape)
print(G[0]) #Gives the samples of the first variant
print(G[:,0]) #Gives the variants of the first sample
print(G.mean(axis=0)) #This gives you mean per variant

g = G[0]
print(g)
avr = g.mean()
print(avr)

print(f"The first step of standardization should look like{[x - avr for x in g]}")

std = [(x - avr)**2 for x in g]
std = math.sqrt(sum(std)/(len(g)-1))
#computed std by hand from scratch

print(std)

#NumPy Verification with Bessel's correction, giving sample std
print(np.std(g, ddof=1))

means = np.array([sum(row)/len(row) for row in G])
print(means.shape)
print(G.shape)
G_fixed = G - means[:,np.newaxis]
stds = (np.sum(G_fixed**2, axis = 1)/(len(G[0])-1))**0.5
print(stds)
print(G_fixed)

G_std = G_fixed/stds[:,np.newaxis]

print(G_std)
print(G_std @ G_std.T/ (len(G[1])-1))

print(np.corrcoef(G))

