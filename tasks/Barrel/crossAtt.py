
#%%
import pickle
import numpy as np
import keras
from keras import layers
import numba
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import confusion_matrix, roc_curve

hep.style.use("CMS")

pkl=pickle.load(open('pickle_data.p', 'rb'))
dfDict=pkl["data"]
varDict=pkl["columns"]
genPt=pkl['genpt']


Tk = dfDict['Tk']
CryClu= dfDict['CryClu']
y=CryClu[:,:,-1]
CryClu=CryClu[:,:,:-1]

weights1=len(y[y!=-999])/(2*np.sum(y.ravel()==1))
weights0=len(y[y!=-999])/(2*np.sum(y.ravel()==0))
class_weights = {0: weights0, 1: weights1}

y[y==-999]=0

#%%

@numba.jit
def build_mask(Tk,CryClu):
    mask=np.ones((Tk.shape[0],CryClu.shape[1],Tk.shape[1]),dtype=numba.types.bool_)
    for ev in range(Tk.shape[0]):
        for c in range(CryClu.shape[1]):
            if CryClu[ev,c,0]==-999:
                mask[ev,c,:]=False
        for t in range(Tk.shape[1]):
            if Tk[ev,t,0]==-999:
                mask[ev,:,t]=False
    return mask
                
mask=build_mask(Tk,CryClu)

#%%

def build_model(dropout=0.05,nunits=32,nlayers=2,activation='swish'):
    CryCluInput = keras.Input(shape=(CryClu.shape[1],CryClu.shape[2],), name='CryCluInput')
    xC = layers.BatchNormalization()(CryCluInput)
    
    for _ in range(nlayers):
        xC = layers.Dense(nunits, activation=activation)(xC)
        xC = layers.Dropout(dropout)(xC)

    TkInput = keras.Input(shape=(Tk.shape[1],Tk.shape[2]), name='TkInput')
    xT = layers.BatchNormalization()(TkInput)
    
    for _ in range(nlayers):
        xT = layers.Dense(nunits, activation=activation)(xT)
        xT = layers.Dropout(dropout)(xT)
    
    maskInput = keras.Input(shape=(CryClu.shape[1],Tk.shape[1]), name='maskInput')
    crossAtt= layers.MultiHeadAttention(num_heads=1,
                                        key_dim=nunits,
                                        value_dim=nunits,
                                        dropout=dropout)(xC, xT, xT,
                                        attention_mask=maskInput,)
                                        
    x= layers.Add()([crossAtt,xC])
    xadd=layers.LayerNormalization()(x)
    
    for i in range(nlayers):
        if i==0:
            inputx=xadd
        else:
            inputx=x
            
        x = layers.Dense(nunits, activation=activation)(inputx)
        x = layers.Dropout(dropout)(x)

    x= layers.Add()([xadd,x])
    x= layers.LayerNormalization()(x)
    x= layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=[CryCluInput, TkInput,maskInput], outputs=x)
    return model


def compile_model(model,
                  learning_rate=0.001,
                  validation_split=0.2,
                  epochs=100,
                  batch_size=64):
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    
    history = model.fit(
        [CryClu, Tk, mask],
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        class_weight=class_weights,)
    return history

def plot_history(history,item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


model=build_model(dropout=0.05)
history=compile_model(model,batch_size=128,epochs=15,learning_rate=0.001,validation_split=0.2)
plot_history(history,'loss')
# %%


nev=len(Tk)
y_pred = model.predict([CryClu[int(0.8*nev):], Tk[int(0.8*nev):], mask[int(0.8*nev):]])
y_test = y[int(0.8*nev):]

#%%

fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred.ravel())
#plot the eddiciecy vs trigger rate for phase 2
plt.plot(fpr*11245.6*2500/1e3, tpr, label='ROC curve')

plt.xlabel('Trigger Rate [kHz]')
plt.ylabel('Electron efficiency')
plt.xlim(0,40)
plt.grid()
hep.cms.text("Phase2 Simulation")
hep.cms.lumitext("PU200 (14 TeV)")


#%%
conf=confusion_matrix(y_test.ravel(),y_pred.ravel()>0.5,normalize='true')
print(conf)
plt.imshow(conf, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0,1],['Background','Signal'])
plt.yticks([0,1],['Background','Signal'])

#%% genPt eff

def get(obj,var):
    idx=varDict[obj].index(var)
    return dfDict[obj][:,:,idx]
    

ptSig=genPt[int(0.8*nev):][y_test==1]
ptSigCorr=ptSig[y_pred[y_test==1,0]>0.5]
ptSigNotCorr=ptSig[y_pred[y_test==1,0]<0.5]

hSig=np.histogram(ptSig,bins=50,range=(0,100))
hSigCorr=np.histogram(ptSigCorr,bins=50,range=(0,100))

plt.hist([ptSigCorr,ptSigNotCorr],bins=50,range=(0,100),label=['Correct','Incorrect'],stacked=True)

plt.grid()
plt.xlabel('genPt [GeV]')
plt.legend()

#%%
plt.figure()
cen=(hSig[1][1:]+hSig[1][:-1])/2
eff=hSigCorr[0]/hSig[0]
plt.step(cen,eff,where='mid',marker='v',label='Electron')
plt.ylabel('Efficiency')
plt.xlabel('genPt [GeV]')
plt.grid()
plt.legend()

# %%
