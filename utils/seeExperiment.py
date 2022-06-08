from utils.LoadExperiment import *

var1 = 'lr'
var2 = 'n_layers'
df = load_exp_result('exp-2022-05-03-0daa54')

plot_acc(var1, var2, df)
plot_loss_variation(var1, var2, df, sharey=False)
plot_acc_variation(var1, var2, df, margin_titles=True, sharey=True)
