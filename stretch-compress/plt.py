import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
rcParams["figure.dpi"] = 300

def average_5points(x, y):
    n = len(x) // 5 * 5
    x_trim = x[:n]
    y_trim = y[:n]
    
    x_5blocks = x_trim.reshape(-1, 5)
    y_5blocks = y_trim.reshape(-1, 5)
    
    y_mean = np.mean(y_5blocks, axis=1)
    x_mid = x_5blocks[:, 1]
    
    return x_mid, y_mean

data1 = np.genfromtxt("./dihedral_auall.txt", names=True)
x1_original = data1['step'] / 1000
y1_original = data1[data1.dtype.names[-1]]

x1_mid, y1_mean = average_5points(x1_original, y1_original)

data2 = np.genfromtxt("./conduct.txt", names=True)
x2_original = data2['step'][:-1] / 1000
y2_original = data2['conduct'][:-1]

x2_mid, y2_mean = average_5points(x2_original, y2_original)

fig, ax1 = plt.subplots(figsize=(11, 6))

ax1.scatter(x1_original, y1_original, color="#0A85D0", s=100,alpha=0.18, label='Sum of Dihedral Angles')

ax2 = ax1.twinx()
ax2.scatter(x2_mid, y2_mean, color="#DA4E51", marker='x', s=150, label='Conductance (avg)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left',prop={'size': 20})
x_major_locator=MultipleLocator(0.1)
y_major_locator=MultipleLocator(90)
ax1.xaxis.set_major_locator(x_major_locator)
ax1.yaxis.set_major_locator(y_major_locator)
plt.xticks(size=16)
ax1.set_xlabel('Time (ns)', fontsize=22) 
ax1.set_ylabel('Sum of Dihedral Angles (°)', color='black', fontsize=22)
ax2.set_ylabel('Conductance (log(G/G$_0$))', color='black', fontsize=22)
ax1.tick_params(labelcolor='black', labelsize=20)
ax2.tick_params(labelcolor='black', labelsize=20)
ax1.set_ylim(630,1350)
ax2.set_ylim(-7,-1.6)
plt.savefig('str-com.png')