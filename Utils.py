#Utilities file for main program

#Imports
from numpy import mean,std,array
from numpy.random import rand
from numpy import exp as np_exp

#Utilitary functions
def normalize(df):
    "Normalize x and y. Set min(y) to 0"""
    m_x,m_y = mean(df.x),mean(df.y)
    s_x,s_y = std(df.x),std(df.y)
    
    dfn = (df.copy())
    dfn.x = (df.x - m_x)/s_x 
    dfn.y = (df.y - m_y)/s_y
    dfmin = min(dfn.y)
    dfn.y -= dfmin
    return dfn.reset_index(drop=1),(m_x,m_y,s_x,s_y)

#Weight function. Useful for reducing amount of data
def weigh(df,prob,Range=(0.23,1.0)):
    """prob = probability of keeping a given point outside of the range specified by Range"""
    from numpy.random import rand
    reduce = df[(df.x<Range[0]) | (df.x>Range[1])].reset_index()
    init_shape = reduce.shape[0]
    
    n_red = df[(df.x>Range[0]) & (df.x<Range[1])]
    for i in range(reduce.shape[0]):
        if rand()>prob:
            reduce.drop(i,inplace=True)

    print("{} points dropped".format(init_shape-reduce.shape[0]))
    return reduce.merge(n_red,how='outer').drop(columns='index').sort_values(by='x')

#Load 1 dataset
from pandas import read_csv
def LoadDF(name,dom=(210,400),weight=(-0.3,1.6,0.3)):
    """name = /path/to/file to load
    dom: Clean data. Cut from t=dom[0] to t=dom[1]. This is determined by plotting t vs T. 
        Default values ar ok for this datasets
    weight: Give importance to data in the range from (-0.3,1.2) by a factor of 0.7"""    
    df = (read_csv(name,encoding='utf-16',
                   skiprows=80,header=None,sep='[ ,\t]',
                   engine='python',names=['t','x','W','H','c','v','f'])
          .astype(float)
          .drop(columns=['H','c','v','f']))

    df = df[(df['t'] > dom[0]) & (df['t'] < dom[1] )].reset_index(drop=True) 
    df['W/%']=df['W']*100.0/df['W'].iloc[0]
    df.columns = ['t','x','W','y']
    
    dfn,norm_prms = normalize(df)
    dfR=weigh(dfn,0.5,Range=(0.0,1e-5))
    dfR=weigh(dfR,weight[2],Range=(weight[0],weight[1]))

    return dfR,norm_prms

#Create custom layer. This layer computes our function. This is done so we can compile a TF model -> easy to train
from tensorflow import constant_initializer
from keras.layers import Input,Dense,Layer
from keras.models import Model,load_model
from keras.optimizers import Adam
from tensorflow.math import exp as tf_exp

def create_model(xps,ks,As,Ss):
    inp = Input(shape=(1,))
    out = MyLayer(xps,ks,As,Ss)(inp)
    model = Model(inputs=inp, outputs=out)
    return model

class MyLayer(Layer):
    def __init__(self,ks,As,Ss,freezeS,**kwargs):
        """freezes = [0,0,0,1,1,0]
        Custom layer creator. This allows the use of TF infrastructure, so we treat our model as a NN"""
        super(MyLayer, self).__init__(**kwargs)
        A = [i for i in range(len(Ss))]
        k = [i for i in range(len(Ss))]
        s = [i for i in range(len(Ss))]

        self.As = As
        self.ks = ks
        self.Ss = Ss
        
        sTrainable = True
        for i in range(len(Ss)):
            A[i] = self.add_weight(name='A{}'.format(i),
                                   shape=(),trainable=True, initializer = constant_initializer(value=As[i]))

            k[i] = self.add_weight(name='k{}'.format(i),
                                   shape=(),trainable=True, initializer = constant_initializer(value=ks[i]))

            if freezeS[i] == 1:
                sTrainable = False
            s[i] = self.add_weight(name='s{}'.format(i),shape=(),
                                   trainable=sTrainable, initializer = constant_initializer(value=Ss[i]))
            sTrainable=True
        
        self.A = A
        self.k = k
        self.s = s
        self.B = self.add_weight(name='B',shape=(),
                                 initializer=constant_initializer(value=-2.08),trainable=True)
    def call(self, x):
        f = self.B
        for i in range(len(self.Ss)):
            f = f + self.A[i]/(1+tf_exp(self.k[i]*(x-self.s[i])))
        return f
    
def ExtractParams(BroutePrms,freezeS):
    """Extract all parameters to a dictionary, based on:
    BroutePrms: List of all trained parameters. BroutePrms = model.layers[1].get_weights()
    freezeS: List that tells as a boolean var. whether S_i was freezed"""
    prms = {}
    #First extract parameters that are not trainable (at end of list)
    nFreez = int(sum(freezeS))
    if nFreez > 0:
        non_train = BroutePrms[-nFreez:]
        trainable = BroutePrms[:-nFreez]
    else:
        non_train = []
        trainable = BroutePrms
    
    j=0
    for i in range(len(freezeS)): #Freezed S vars
        if freezeS[i] == 1:
            prms[f's{i}'] = non_train[j]
            j+=1

    #Now extract all non-trainable parameters
    """Start at j=2. If next param was non-trainable, say s3, of course you don't count it.
    Insted you jump to next weight (A4 in this case)"""
    j = 2  
    for i in range(len(freezeS)):
        prms[f'A{i}'] = trainable[3*i + j-2]
        prms[f'k{i}'] = trainable[3*i + j-1]
        if freezeS[i] == 1:
            #If S_i is freezed, count differently since it is not in this list
            j-=1
        else:
            #Only assign S_i a value if it is not freezed.
            prms[f's{i}'] = trainable[3*i + j]
    prms['B'] = trainable[-1]  
    return prms


     
#Cost function
from tensorflow import cond,boolean_mask
from tensorflow.math import abs as tf_abs
from tensorflow.math import multiply as tf_multiply
from tensorflow.math import square
from tensorflow.keras.losses import MSE
from numpy import vectorize,float32

def CostFunction(prms,hyper,peak):        
    """Base cost function. 
    Any modification to this function may be written here directly 
    or by copying this function into the given notebook and editing there.
    
    Next versions should leave this function as a real base cost, additional restrictions imported as other functions."""
    Ss,dev_pts,posit,ks,As,freezeS = hyper
    def loss(y_true,y_pred):
        
        A,k,S,B = prms
        #Set S values to lie within a range given by parameters
        def S_cost(Ss,dev_pts): 
            Cost=0
            for i in range(len(Ss)):
                dif_sq = (S[i] - Ss[i])**2
                Cost+=cond(dif_sq > dev_pts[i], lambda: 30*dif_sq,lambda: 0.)
            return Cost
    
        #Set each function to have a specific orientation
        def sign_cost(Ss,posit):
            Cost=0
            for i in range(len(Ss)):
                Cost+=cond(A[i]*k[i]*posit[i] > 0.,
                           lambda: tf_abs(A[i]),lambda: 0.)
            return Cost
        
        def ChemicalRequire(Peak,Ss):
            """Add chemical requirements: 
            Req. 1: The 3 first curves are CO2 absorption, 4th and 5th correspond to loss of it.
            Req. 2: First curve should start at 0.
            Next versions: This should be modified so user can add this requirements without having to look up at this code
            """
            ########## All of the below code is hard-coded.
            #Sum all N-3 curves (all but 3 last).
            First3Sum = sum([tf_abs(A[i]) for i in range(len(posit))]) - sum([tf_abs(A[-i]) for i in range(1,4)])
            
            #TotalSum - Last3Sum = Peak
            Diff = tf_abs(First3Sum - Peak) 
            #Next 2 curves also equal Peak
            Dif2 = tf_abs(A[-2]+A[-3] - Peak)
            
            Req1Cost = cond(Diff+Dif2 > 1e-6, lambda: 50.0*(Diff+Dif2),lambda: 0.)
            ########## All of the above code is hard-coded. Fix for next versions
            
            #Req 2: First curve starts at 0. This is achieved summing all Ai such that ki>0, + B = 0
            SumOffA = B 
            for i in range(len(Ss)):
                SumOffA += A[i]*cond(k[i] > 0.,lambda: 1.,lambda: 0.)
            Req2Cost = cond(tf_abs(SumOffA) > 1e-3, lambda: 10.0*tf_abs(SumOffA),lambda: 0.)
            
            return  Req1Cost + Req2Cost

        return (MSE(y_true,y_pred)+
                S_cost(Ss,dev_pts)+
                sign_cost(Ss,posit)+
                1e-3*ChemicalRequire(peak,Ss))
    return loss

from math import isnan
from keras import callbacks
from tensorflow.compat.v1 import Session, global_variables_initializer
from tensorflow.train import Checkpoint,latest_checkpoint
from keras import callbacks

class EndOnNaN(callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        if isnan(loss):
            print("\n\n############\nModel has failed to converge!\
                  \nRestart optimization\
                  \nTry lowering lr.\
                  \n#############\n")
            self.model.stop_training = True
            
            
def Optimize(df,hyper,peak,lr=5e-2,epochs=100,v=1,weights=0):
    """df: training data
    hyper: Matrix with hyperparameters. Contains, as rows: Ss,dev,posit,ks,As,freezeS
    Ss: x point around which curves are to be centered
    dev: allowed deviation for the center of the curves with respect to Ss
    posit: slope of the curve: slope==1 -> positive slope, slope==0 -> negative slope
    ks: Suggested k values (steepness of curve)
    As: Suggested A values (relative importance of event, by how much did mass increment with this event?)
    freezeS: Which S values are to be fixed
    lr: learning rate
    epochs: number of epochs
    v: verbosity of output while training
    weights: load pre-trained parameters to restart optimization from there"""
    
    sess=Session()
    with sess.as_default():
        with sess.graph.as_default():
            Ss=hyper[0,:]
            ks = hyper[3,:]
            As = hyper[4,:]
            freezeS = hyper[5,:]
            
            model=create_model(ks,As,Ss,freezeS)
            prms=[model.layers[1].A,
                  model.layers[1].k,
                  model.layers[1].s,
                  model.layers[1].B]
            model.compile(loss=CostFunction(prms,hyper,peak=peak),
                          optimizer=Adam(lr))

            if weights:  #Load weights if available
                model.load_weights(weights)

            #Callbacks
            rlr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,patience=20,min_delta=0.004,
                                                  min_lr=2.0e-7, mode='min', verbose=v)
            ton = EndOnNaN()
            chk = callbacks.ModelCheckpoint('/tmp/W.hdf5', monitor='loss',save_best_only=True,
                                                mode='min', period=1)
            earlyStp = callbacks.EarlyStopping(monitor='loss', min_delta=0.000003, 
                                                   patience=50, verbose=1, mode='min')

            hist=model.fit(df.x,df.y,epochs=epochs, batch_size=128,verbose=v,
                           callbacks=[rlr,ton,chk,earlyStp])

            model.load_weights('/tmp/W.hdf5') #Load weights (best of all epochs)

            #Extract optimized weights into a dictionary
            BroutePrms = model.layers[1].get_weights()
            prms = ExtractParams(BroutePrms,freezeS)
            return prms,min(hist.history['loss'])
        
from numpy import ones,zeros
def LoadModel(weights,hyper):
    """Loads a pre-trained model with n curves located at weights = path/to/weights.hdf5"""
    sess=Session()
    with sess.as_default():
        with sess.graph.as_default():
            Ss = hyper[0,:]
            ks = hyper[3,:]
            As = hyper[4,:]
            freezeS = hyper[5,:]
            
            model=create_model(ks,As,Ss,freezeS)
            prms=[model.layers[1].A,
                  model.layers[1].k,
                  model.layers[1].s,
                  model.layers[1].B]
            model.compile(optimizer=Adam(),loss='mse')
            model.load_weights(weights)
            
            #Extract optimized weights into a dictionary
            BroutePrms = model.layers[1].get_weights()
            prms = ExtractParams(BroutePrms,freezeS)
            return prms

from scipy.signal import find_peaks
from pandas import read_csv
from numpy import convolve,ones,diff

def TransferHyper(df,hyper,CurvePeak = [1,3]):
    """Returns the whole "hyper" array based on an initial model,
    transforming it according to the derivatives of this new dataset
    IMPORTANT! CurvePeak: Which curves are responsible for the LARGEST peak? Drop here 2 indices.
    For La0.9, CurvePeak = [1,3] but for others it might be [1,4]"""
    def smooth(y, box_pts=30):
        box = ones(box_pts)/box_pts
        y_smooth = convolve(y, box, mode='same')
        return y_smooth
    deriv = diff(df.y)/diff(df.x)
    deriv = smooth(deriv,50)
    
    peak1 = find_peaks(deriv,height=5)
    pkX1 = df.x[peak1[0][peak1[1]['peak_heights'].argmax()]]  #Get x-value of first (up) peak
    
    peak2 = find_peaks(-deriv,height=10,distance=50)
    index = array(sorted(zip(peak2[1]['peak_heights'],peak2[0]),key=lambda x : x[0])[-2:])[:,1]
    pkX2 = sorted(df.x[index].values)  #Get x-values of doublet peaks

    Svals = hyper[0,:]
    
    #Reescale the other peaks. Bring them to feasible values given those for the new peaks.
    x1,x2 = pkX1,pkX2[-1]
    m = (x2-x1)/(Svals[CurvePeak[1]]-Svals[CurvePeak[0]])  #Mult. constant
    b = x2 - m * Svals[CurvePeak[1]]   #Sum. constant
    newS = []
    for xi in Svals:
        newS.append(m*xi + b)
        
    #Set new S value
    hyper[0,:] = newS
    
    for i in range(hyper.shape[1]):
        ki = hyper[3,i]
        if ki > 60.:
            #Rescale k vals to ensure numerical stability
            hyper[3,i] = hyper[3,i]*0.75 
    return hyper.copy()

def BuildHyperFromModel(prms,hyper):
    """N = number of curves.
    This function builds the array "hyper" based on the weights loaded from a previously trained model."""
    #Modify only S,k,A. Other entries of hyper are left the same as for the first model
    for i in range(hyper.shape[1]):
        hyper[0,i] = prms[f's{i}']
        hyper[3,i] = prms[f'k{i}']
        hyper[4,i] = prms[f'A{i}']
    return hyper


from numpy import log

def FeatureDict(PRMS,names,dfs,steqs=['0.9'],exponent = 2):
    X = {}  #Each dict will contain one dict for each Steq
    Y = {}

    #So you will end up indexing like this:  X[Steq][s1] gives all x values for event 1 on this stoichiometry
    for Steq in steqs:
        X[Steq],Y[Steq] = {},{}  #For each Steq, each dict will contain a list for each event (s1,s2...)
        for i in range(6):
            X[Steq][f's{i}'],Y[Steq][f's{i}'] = [],[]

        for name in names:
            CompoundName = name.split('/')[-1]
            st = CompoundName[2:5]  #steq
            if st==Steq:
                B = float(CompoundName.split('.')[2].split('NiO3')[1][:-1])  #Heat speed
                prms = PRMS[CompoundName]
                df,hyper,normPrs = dfs[name]

                for i in range(6):
                    T1 = prms[f's{i}'] * normPrs[2] + normPrs[0] + 273.15 #Renormalize and convert to Kelvin
                    X[Steq][f's{i}'].append(1/T1)
                    Y[Steq][f's{i}'].append(log(B/(T1**exponent)))
    return X,Y


from scipy.stats import linregress
from numpy import poly1d,linspace

def PlotLinearModels(X,Y,save=False,xlims=False):
    """If save!=False, it must contain the name with which the plot is to be saved.
    xlims is a tuple containing x limits of the plot"""
    steqs = list(X.keys())
    fig,ax = subplots(len(steqs),figsize=(10,6*len(steqs)),sharex=True)
    rcParams['font.size'] = 15
    colors = ['b','orange','green','red','purple','k']
    
    if len(steqs) == 1:
        ax = [ax]
        
    for j in range(len(steqs)):
        steq = steqs[j]
        ax[j].set_ylabel(r'ln(B/T$^2)$')
        for i in range(6):
            x = X[steq][f's{i}']
            y = Y[steq][f's{i}']
            lr = linregress(x,y)
            f = poly1d(lr[:2])
            M,I = lr[:2]
            R = lr.rvalue**2
            ax[j].plot(x,y,'x',color=colors[i])
            xs = linspace(mean(x)-2*std(x),mean(x)+2*std(x))
            ax[j].plot(xs,f(xs),'--',color=colors[i], label = f'y = {M:.4E} x + {I:.2f}     R={R:.2f}')
        
        if xlims:
            ax[j].set_xlim(xlims[0],xlims[1])
        ax[j].legend(fontsize=10)
        ax[j].grid()

    for i in range(len(steqs)):
        ax[i].set_title(f'La{steqs[i]}Ca{round(1-float(steqs[i]),2)}NiO3')

    ax[-1].set_xlabel('1/T')
    tight_layout()
    if save:
        savefig(save,dpi=300)
    show()

    
#Plot functions. This will need modifications
from numpy import diff
def plot_derivs(df,prms,n):
    deriv = diff(df.y)/diff(df.x)
    
    fig,ax=subplots(figsize=(15,10))
    ax.plot(df.x[:-1],deriv,'k-',linewidth=1)
    
    ax.set_xlabel('T/C')
    ax.set_ylabel('Weight derivative/ % T$^{-1}$')
    pred_deriv = 0
    for i in range(n):
        A,k,s = prms[f'A{i}'],prms[f'k{i}'],prms[f's{i}']      
        
        expc = np_exp(k*(df.x-s)) #Exponential term
        func_i = -A*k*expc/(expc+1)**2
        pred_deriv += func_i
        ax.plot(df.x,func_i,'-.')
    
    ax.grid()
    #ax.set_xlim(300,1000)
    #ax.set_ylim(-1,0.3)
    #ax.plot(df.x,pred_deriv,'-',color='lime')
    #savefig('Deriv.png',dpi=300)
    show()
    
from matplotlib.pyplot import subplots,plot,close,tight_layout,show,rcParams,savefig
from numpy.random import rand
from numpy import exp as np_exp
def plot_f(df,prms,pts): 
    def f(prms,X,n):
        func = prms['B']
        for i in range(n):
            A,k,s = prms[f'A{i}'],prms[f'k{i}'],prms[f's{i}']
            func += A/(1+np_exp(k*(X-s)))
        return func

    n = pts.shape[1]  
    pred=f(prms,array(df.x),n).flatten()

    fig,ax=subplots(figsize=(15,10))
    ax.plot(df.x,pred,'r-')
    ax.plot(df.x,df.y,'k--')
    
    for i in range(n):
        A,k,s = prms[f'A{i}'],prms[f'k{i}'],prms[f's{i}']    
        func_i = A/(1+np_exp(k*(df.x-s))) + pts[1,i]
        #The last term moves the funtion up or down to match the chosen center
        ax.plot(df.x,func_i,'--')
    show()
    
from numpy import vstack
def plot_wr(df,prms,hyper): 
    """Plot the individual curves scaled to visually represent the restrictions"""
    nCurves=hyper.shape[1]
    NewYPts = [i for i in range(nCurves)]
   
    for i in range(nCurves):
        Sum=0
        if prms[f'A{i}'] < 0:               #If A<0 then sum A to the curve, so the tail initially starts at 0.
            Sum+=abs(prms[f'A{i}'])
        if i<nCurves-3:
            for j in range(i):
                Sum+=abs(prms[f'A{j}'])     #For first nCurves-3 curves: sum full amps. of curves before them. 

        if i == 3:
            Sum+=abs(prms['A4'])
        
        NewYPts[i]=Sum
    
    pts = vstack([hyper[0,:],NewYPts])
    plot_f(df,prms,pts)
