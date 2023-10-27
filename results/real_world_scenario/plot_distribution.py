import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import seaborn as sns


with open('total_gen_smiles.txt') as f:
    lines = f.readlines()

value_list = []
for line in lines:
    value = float(line.strip().split()[-1])
    value_list.append(value)

#plt environment
label_fontsize = 20
tick_length = 6
tick_width = 1.5
tick_labelsize = 16
legend_fontsize = 16


sns.histplot(value_list)
plt.xlabel('TADF-likeness', fontsize=label_fontsize)
plt.ylabel(f'Count', fontsize=label_fontsize, color='k')
plt.tick_params(length=tick_length, width=tick_width, labelsize=tick_labelsize, labelcolor='k', color='k')
plt.xticks([-40, -20, 0, 20, 40, 60, 80, 100])
plt.show()
