#%%
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import mplhep as hep
import pickle
hep.set_style(hep.style.CMS)


dfDict=pickle.load(open('pickle_data.p', 'rb'))["data"]
varDict=pickle.load(open('pickle_data.p', 'rb'))["columns"]

#%%
cc=dfDict["CryClu"]
sh=cc.shape
cc=cc.reshape(sh[0]*sh[1],sh[2])
cc=cc[cc[:,0]!=-999]

sig=cc[cc[:,-1]==1][:,:-1]
bkg=cc[cc[:,-1]==0][:,:-1]


fig=corner.corner(sig,labels=varDict["CryClu"][:-1],levels=(0.5,0.9, 0.99),  color='tab:blue', scale_hist=True,plot_density=True,bins=30)
corner.corner(bkg[:len(sig)],labels=varDict["CryClu"][:-1],levels=(0.5,0.9, 0.99), fig=fig, color='tab:orange', scale_hist=True,bins=30,plot_density=True)
blue_line = mlines.Line2D([], [], color='tab:blue', label='Signal')
red_line = mlines.Line2D([], [], color='tab:orange', label='Background')
plt.legend(handles=[blue_line,red_line],loc='upper right',frameon=False,bbox_to_anchor=(1,5),fontsize=25)
plt.savefig('fig/corner.png',bbox_inches='tight',dpi=300)

#%%

sig=sig[sig[:,0]==1][:,1:]
bkg=bkg[bkg[:,0]==1][:,1:]
#%%

fig=corner.corner(sig,labels=varDict["CryClu"][1:-1],levels=(0.5,0.9, 0.99),  color='tab:blue', scale_hist=True,plot_density=True,bins=30)
corner.corner(bkg[:len(sig)],labels=varDict["CryClu"][1:-1],levels=(0.5,0.9, 0.99), fig=fig, color='tab:orange', scale_hist=True,bins=30,plot_density=True)
blue_line = mlines.Line2D([], [], color='tab:blue', label='Signal')
red_line = mlines.Line2D([], [], color='tab:orange', label='Background')
plt.legend(handles=[blue_line,red_line],loc='upper right',frameon=False,bbox_to_anchor=(1,5),fontsize=25)
plt.savefig('fig/standalone.png',bbox_inches='tight',dpi=300)