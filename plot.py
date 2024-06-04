import matplotlib as mpl
import matplotlib.font_manager as fm
# fm._rebuild()
import matplotlib.pyplot as plt
import numpy as np
# set font size and style
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Times New Roman"
ff  = [10,100,100,100,100,100,100,100,100]
data_samples = [10,20,30,40,50,60,70,80,90]

ff_acc = np.array(ff)
data_samples = np.array(data_samples)
plt.figure(dpi=250, figsize=(4, 3))

plt.plot(data_samples, ff_acc, label='Ours', c='C0', marker='o')
# plt.plot(num_params, brn_loss, label='Block Reversible', c='C1', marker='s')
# plt.xlim(0, 100)
# plt.ylim(0, 110)
plt.legend()
plt.title('Operation: +')
plt.xlabel('Train Data Fraction (%)')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.tight_layout(pad=0)
plt.savefig('id_vs_ood_accs.pdf', bbox_inches='tight')
plt.show()


