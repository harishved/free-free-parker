import matplotlib.pyplot as plt
import numpy as np
#
f = open("gffu.dat")
lines = f.readlines()

u=[]
g2=[]
gff=[]

for line in lines:
   if line[0]=="#":
      continue;
   w = line.strip("\n")
   u.append(float(w[0:15]))
   g2.append(float(w[15:29]))
   gff.append(float(w[29:]))

u=np.array(u)
g2=np.array(g2)
gff=np.array(gff)

Ng2 = int(np.where(np.diff(u))[0][0]+1)
Nu = int(len(gff)/Ng2)

G = np.reshape(gff,(Nu,Ng2))
uvec = u[::Ng2]
g2vec = g2[:Ng2]


print (G.shape)


np.savez("gffu.npz",u=uvec,g2=g2vec,gff=G)


plt.plot(g2vec,G[20,:])

plt.show()
plt.close()






