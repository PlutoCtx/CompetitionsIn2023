import numpy as np
import matplotlib
import matplotlib.pyplot as plt

vegetables = [ 'cdp', 'ma', 'arbr', 'cr',
             'psy', 'obv', 'pvt','bias','mtm','boll']
farmers =  [ 'cdp', 'ma', 'arbr', 'cr',
             'psy', 'obv', 'pvt','bias','mtm','boll']

harvest = np.array([[1,0.14,0.13,0.19,0.27,0.31,0.29,0.16,0.07,0.6],
                    [0.14,1,0.29,0.21,0.05,0.11,0.09,0.07,0.51,0.07],
                    [0.13,0.29,1,0.12,0.16,0.13,0.08,0.11,0.15,0.08],
                    [0.19,0.21,0.12,1,0.7,0.11,0.14,0.16,0.11,0.14],
                    [0.27,0.05,0.16,0.7,1,0.07,0.08,0.11,0.12,0.11],
                    [0.31,0.11,0.13,0.11,0.07,1,0.61,0.64,0.11,0.094],
                    [0.29,0.09,0.08,0.14,0.08,0.61,1,0.71,0.07,0.11],
                    [0.16,0.07,0.11,0.16,0.11,0.64,0.71,1,0.13,0.14],
                    [0.07,0.51,0.15,0.11,0.12,0.11,0.07,0.13,1,0.09],
                    [0.6,0.07,0.08,0.14,0.11,0.094,0.11,0.14,0.09,1]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(farmers)))
ax.set_xticklabels(farmers)
ax.set_yticks(np.arange(len(vegetables)))
ax.set_yticklabels(vegetables)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Technical factor relation matrix")
fig.tight_layout()
plt.show()