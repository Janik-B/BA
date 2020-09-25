import matplotlib.pyplot as plt
import os
import numpy as np
import csv

dir_name = '../neural_network/data/estimator_vs_reality'
diff_norms = []
diff_smooth = []
norm = 0.5
lr = 0.25
loss = []
csv_path = os.path.join(dir_name, 'estimator_vs_reality.csv'.format(norm, lr))
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        g = float(row['grad_norm'])
        s = float(row['smoothness'])
        g_ex = float(row['grad_norm_exact'])
        s_ex = float(row['smoothness_exact'])
        diff_norms.append(abs(np.math.log(g) - np.math.log(g_ex)))
        diff_smooth.append(abs(np.math.log(s) - np.math.log(s_ex)))
fig, axs = plt.subplots(1, 2)

axs[0].plot(range(len(diff_norms)), diff_norms)
axs[0].set_ylabel("Abweichung von der Norm des Gradienten")
axs[0].set_xlabel("Iteration")
axs[0].set_title("Schätzer für die Norm des Gradienten")

axs[1].plot(range(len(diff_smooth)), diff_smooth)
axs[1].set_ylabel("Abweichung von der Glattheit")
axs[1].set_xlabel("Iteration")
axs[1].set_title("Schätzer für die lokale Glattheit")
plt.show()
