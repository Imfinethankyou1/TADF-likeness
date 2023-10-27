import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

label_fontsize = 10
tick_length = 6
tick_width = 1.5
tick_labelsize = 10
legend_fontsize = 10


values = ["Low", "Medium-low", "Medium-high", "High"] + ["TADF"]
labels = ["HOMO (eV)", "LUMO (eV)", "$E(S_{1})$", "$\Delta E_{ST}$"]
colors = ["y", "orange", "r", "purple", "black"]

fns = [f'../TADF-likeness_test_clustering_{i}.txt' for i in range(5)]
fns += [f'../TADF-likeness_test_clustering_val_top_{i}.txt' for i in range(2,6)]


labels = [f'CV$_{i}$' for i in range(5)]
labels += [f'En$_{i}$' for i in range(2,6)]


data_list = []
for fn in fns:
    data = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        score = float(line.strip().split()[2])
        data.append(score)
    data_list.append(data)
#for i in range(len(fns)):
# ax = plt.subplot(2, 2, i + 1)
#sample_props = [data1, data2, data3, data4, data5]

fig = plt.figure(figsize=(8.0,4.0))

box = plt.boxplot(data_list, notch=True, patch_artist=True, labels=labels, whis=2.5)
# colors = ["cyan", "lightblue", "lightgreen", "tan", "pink", "ivory"]
colors = ["red", "blue", "green", "purple", "orange", "pink", "brown", "grey", "cyan", "lime"][:-1]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
plt.xlabel("Model", fontsize=label_fontsize)
plt.ylabel(f"TADF-likeness", fontsize=label_fontsize, color="k")
plt.tick_params(
    length=tick_length,
    width=tick_width,
    labelsize=tick_labelsize,
    labelcolor="k",
    color="k",
)
plt.tight_layout()
plt.savefig('Figure_S1.pdf')
plt.show()
