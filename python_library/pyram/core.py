#  LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import scipy

from sklearn.linear_model import LinearRegression
from itertools import combinations

from scipy.signal import savgol_filter
from scipy.optimize import linprog
from scipy.spatial import distance
from scipy.optimize import minimize

##################################################################
######################## GLOBAL VARIABLES ########################
##################################################################
module_path = os.path.dirname(__file__)
lib_path = module_path+'/RamanLib'

lib_files, lib_names, lib = (0,0,0)

legend_num=0


###################################################################
######################## FUNZIONI GENERALI ########################
###################################################################

def cwd(dir):
    os.chdir(dir)

def ls():
    print(os.listdir())

def type2spectra(S, interp = True):
    if type(S)==str:
        if S[0:3]=='txt':
            S = loadtxt(txt_alias[S], interp)
        elif S[0:3]=='lib':
            S = lib[lib_alias[S]]
        elif S[0:3]=='sch':
            S = lib[sch_alias[S]]
        else:
            S = loadtxt(S, interp)
    elif type(S)==np.ndarray:
        S = S
    elif type(S)==list:
        for i in range(len(S)):
            if type(S[i])==str:
                if S[i][0:3]=='txt':
                    S[i] = loadtxt(txt_alias[S[i]], interp)
                elif S[i][0:3]=='lib':
                    S[i] = lib[lib_alias[S[i]]]
                elif S[i][0:3]=='sch':
                    S[i] = lib[sch_alias[S[i]]]
                else:
                    S[i] = loadtxt(S[i], interp)
            elif type(S[i])==np.ndarray:
                S[i] = S[i]
            else:
                print('Type error!')
    else:
        print('Type error!')
        return 
    return S

def loadtxt(file, interp=True):
    S = np.transpose(np.loadtxt(file))
    if interp==True:
        S = interpol(S)
    return S

def savetxt(file, S):
    np.savetxt(file, np.transpose(S))


txt_alias = 'Run set_alias() to generate an alias dictionary of the txt files in the current position'
def set_alias():
    files = [j for j in os.listdir() if (j[-4:]=='.txt' and j[0:6]!='README')]
    global txt_alias
    txt_alias = ['txt'+str(j) for j in range(len(files))]
    txt_alias = dict(zip(txt_alias,files))
    for i in txt_alias.keys():
        print(i, ' -> ', txt_alias[i])


#############################################################
######################## MERGE FILES ########################
#############################################################

def merge_spectra(folder, plot=False):
    '''
    Merge the spectra (file .txt) contained in the selected folder
    '''
    # CERCO I FILE NELLLA CARTELLA folder E LI RAGRUPPO
    dir = sorted(os.listdir(folder))
    name = ''
    files=[]
    n_file=-1
    for rep in dir:
        if rep[-4:]=='.txt':
            if rep[:-6]!=name:
                name = rep[:-6]
                files.append([rep])
                n_file +=1
            else:
                files[n_file].append(rep)

    for i in range(len(files)):
        # CARICO TUTTI GLI SPETTRI DI UNA RIPETIZIONE IN UNA LISTA
        spectras_raw=[]
        for j in files[i]:
            spectras_raw.append(loadtxt(folder+'/'+j))

        spectras = copy.deepcopy(spectras_raw)

        for a in range(len(spectras)-1):
            b = a+1
            # CALCOLO REGIONE COMUNE AGLI SPETTRI A E B
            na,nb = (0,0)
            while spectras[a][0][na]<spectras[b][0][0]:
                na+=1
            while spectras[b][0][nb]<spectras[a][0][-1]:
                nb+=1

            # CALCOLO OFFSET A-B
            offset = np.mean(spectras[a][1][na:]) - np.mean(spectras[b][1][:nb])

            # AGGIUNGO OFFSET A SPETTRO B
            spectras[b][1] += offset

        merged = np.concatenate(spectras, axis=1)

        # RESORT WAVELENGHTS
        sorted_index = merged[0].argsort()
        merged = np.array([np.take_along_axis(merged[0], sorted_index, axis=0),np.take_along_axis(merged[1], sorted_index, axis=0)]) 


        if plot==True:
            _raman_labels()
            plt.plot(merged[0], merged[1])
            for a in range(len(spectras_raw)):
                plt.plot(spectras_raw[a][0],spectras_raw[a][1])

            plt.title(files[i][0][0:-7])
            plt.show()
            
        savetxt(files[i][0][0:-7]+'.txt', merged)


#######################################################
######################## PLOTS ########################
#######################################################

def _raman_labels():
    plt.figure(figsize=(10,5))
    plt.xlabel(r'Raman shift [$cm^{-1}$]')
    plt.ylabel('Counts')

def plot(S, norm=True):
    S = type2spectra(S)

    _raman_labels()
    global legend_num
    legend_num = 0
    if type(S)!=list:
        N = np.max(S[1]) if norm else 1
        plt.plot(S[0], S[1]/N, label=str(legend_num))
        legend_num += 1
    else:
        for s in S:
            N = np.max(s[1]) if norm else 1
            plt.plot(s[0], s[1]/N, label=str(legend_num))
            legend_num += 1
    plt.legend()

def replot(S, norm=True):
    S = type2spectra(S)
    global legend_num
    if type(S)!=list:
        N = np.max(S[1]) if norm else 1
        plt.plot(S[0], S[1]/N, label = str(legend_num))
        legend_num += 1
    else:
        for s in S:
            N = np.max(s[1]) if norm else 1
            plt.plot(s[0], s[1]/N, label=str(legend_num))
            legend_num += 1
    plt.legend()
    plt.draw()


###################################################################
######################## WORK WITH SPECTRA ########################
###################################################################

def interpol(S):
    min = int(S[0][0]) if (S[0][0]-int(S[0][0]))==0 else int(S[0][0]) +1
    max = int(S[0][-1])
    X = np.arange(min,max+1,1)
    S = np.array([X, np.interp(X,S[0],S[1])])
    return S


#################################################################
######################## BKG SUBTRACTION ########################
#################################################################


def bkg_subtraction(S, L_n=31, sigma=150, p=100, edge_width=5, edge_weight=5, plot=False, return_bkg=False):

    #type2spectra
    S = type2spectra(S)
    S = interpol(S)

    # smoothing
    S_smooth = S.copy()
    S_smooth[1]=savgol_filter(S[1],L_n,1)

    # matrice X
    X = np.zeros((len(S_smooth[0]),p))
    for i in enumerate(np.linspace(S_smooth[0,0]-sigma,S_smooth[0,-1]+sigma,p)):
       X[:, i[0]] = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(S_smooth[0]-i[1])**(2)/(2*sigma**2))
   
    # vector C
    H = np.concatenate((np.ones(edge_width)*edge_weight, np.ones(X.shape[0]-2*edge_width), np.ones(edge_width)*edge_weight))
    C = - np.dot(H,X)

    # solving the linear programming problem
    res = linprog(C, A_ub=X, b_ub = S_smooth[1], bounds = (0, None), method='highs-ds')
    W = res['x']
    status = res['status'] # status = 0 means OK
    print('sum(Y-XW) = ', res['fun'])
    print('message = ', res['message'])
    print('# of iter = ', res['nit'])


    if status!=0:
        print('Fit failed !!!')
        return
    # generate the BKG

    BKG = np.array([S_smooth[0], np.dot(X,W)])
    S_nobkg = S.copy()
    S_nobkg[1] -= BKG[1]

    if plot==True:
        _raman_labels()
        plt.plot(BKG[0],BKG[1], label='Estimated background')
        plt.plot(S[0],S[1], label='Original signal')
        plt.plot(S_smooth[0],S_smooth[1], label='Smoothed signal')
        XX = np.arange(200,1000)
        plt.plot(XX, S[1,100]/2*(np.exp(-(XX-600)**(2)/(2*sigma**2))),label='Kernel function')
        plt.legend()
        plt.show(block=False)
    
    if return_bkg:
        return S_nobkg, BKG
    else:
        return S_nobkg

###################################################################
######################## Raman Library ############################
###################################################################


def charge_lib(database='short'):
    '''
    Available databases: short or full
    '''

    print('Charging pure spectra library, needs around a minute ....')

    global lib_files
    global lib_names
    global lib
    lib_files = os.listdir(lib_path + '/' + database)
    lib_names = []

    if database=='full':
        for file in lib_files:
            name=''
            n=0
            for i in file:
                if i=='_':
                    n += 1
                if n<7:
                    name += i
            lib_names.append(name)
    if database=='short':
        for file in lib_files:
            lib_names.append(file[:-4])
        

    lib = []
    for file in lib_files:
        lib.append(np.transpose(np.loadtxt(lib_path+'/'+database+'/'+file)))

    lib = dict(zip(lib_names,lib))
    print('Done. Good work!')
    return

def lib(lib_name=None):
    if lib_name==None:
        return lib
    else:
        return(lib[lib_name])


lib_alias= 'Run lib_names(...) to generate an alias dictionary of the library spectra in the current position'
def lib_names(lib_name=None):
    # ricerca delle iniziali
    temp=[]
    for i in lib_names:
        if i[0:len(lib_name)]==lib_name:
            temp.append(i)
    
    #genero tabella e alias
    global lib_alias
    lib_alias = ['lib'+str(j) for j in range(len(temp))]
    lib_alias = dict(zip(lib_alias,temp))

    print(pd.DataFrame({'alias':lib_alias.keys(), 'name':lib_alias.values()}))
    return


def cos_sim(A,B):
    # portare due spettri alla stessa lunghezza
    min = np.max([A[0,0],B[0,0]])
    max = np.min([A[0,-1],B[0,-1]])
    min_A = np.where(A[0]==min)[0][0]
    max_A = np.where(A[0]==max)[0][0]
    min_B = np.where(B[0]==min)[0][0]
    max_B = np.where(B[0]==max)[0][0]

    A = A[:,min_A:max_A]
    B = B[:,min_B:max_B]

    # calcolo la similarità
    return np.dot(A[1],B[1]) / np.sqrt(np.dot(A[1],A[1])*np.dot(B[1],B[1]))

def cos_sim_w_shift(A,B,shift):
    match_shift = []
    for i in np.arange(-shift,shift+1):
        B_temp = B.copy()
        B_temp[0] += i
        match_shift.append(cos_sim(A,B_temp)) 
    return np.max(match_shift)

def sigmoid(x,s=1):
    return 1 / (1 + np.exp(-s*x))

def preprocessing(P, pre=False):
    if pre==True:
        P[1] = np.where(P[1]>0, P[1], P[1]*0)

        P[1] /= np.max(P[1])
        #P[1] = sigmoid(P[1],10)-0.5
        P[1]=P[1]**(2)
    else:
        P[1] = np.where(P[1]>0, P[1], P[1]*0)
    return P


sch_alias = 'Run search(...) to generate an alias dictionary of the library spectra in the current position'
def search(S, shift=5, first=10, pre=False, verbose=True):
    S = interpol(S)
    S =preprocessing(S,pre)
    match=[]
    for i in lib_names:
        B=lib[i].copy()
        B = preprocessing(B,pre)

        match.append(cos_sim_w_shift(S,B,shift))        

    #genero tabella e alias
    match = pd.DataFrame({'name':lib_names, 'match': match})
    match.sort_values('match', ascending=False, inplace=True)
    match.reset_index(inplace=True, drop=True)

    global sch_alias
    sch_alias = ['sch'+str(j) for j in range(len(match.name))]
    sch_alias = dict(zip(sch_alias,match.name))
    match.insert(0, 'alias', list(sch_alias.keys()))
    if verbose:
        print(match.head(first))
    return match


###################################################################
######################### Read files ##############################
###################################################################


#########   OMNIC FILES   #########

def read_omnic_spa(filepath):
    '''
    Input
    Read a file (string) *.spa
    ----------
    Output
    Return spectra, wavelenght (nm), titles
    '''
    with open(filepath, 'rb') as f:
        f.seek(564)
        Spectrum_Pts = np.fromfile(f, np.int32,1)[0]
        f.seek(30)
        SpectraTitles = np.fromfile(f, np.uint8,255)
        SpectraTitles = ''.join([chr(x) for x in SpectraTitles if x!=0])

        f.seek(576)
        Max_Wavenum=np.fromfile(f, np.single, 1)[0]
        Min_Wavenum=np.fromfile(f, np.single, 1)[0]
        # print(Min_Wavenum, Max_Wavenum, Spectrum_Pts)
        Wavenumbers = np.flip(np.linspace(Min_Wavenum, Max_Wavenum, Spectrum_Pts))

        f.seek(288);

        Flag=0
        while Flag != 3:
            Flag = np.fromfile(f, np.uint16, 1)

        DataPosition=np.fromfile(f,np.uint16, 1)
        f.seek(DataPosition[0])

        Spectra = np.fromfile(f, np.single, Spectrum_Pts)

    # print information
    print('INFO:')
    print('    file name = ', filepath)
    print('    Spectra title = ', SpectraTitles)
    print('    from %f to %f, step = %f, # of channels = %d'%(Min_Wavenum, Max_Wavenum, Wavenumbers[1]-Wavenumbers[0], len(Wavenumbers)))
    return np.array([Wavenumbers[::-1],Spectra[::-1]])


def read_omnic_map(omnic_file):

    omnic_file = OmnicMap.OmnicMap(omnic_file)

    wn_min = omnic_file.info['OmnicInfo']['First X value']
    wn_max = omnic_file.info['OmnicInfo']['Last X value']
    wn_step = omnic_file.info['OmnicInfo']['Data spacing']
    N_wn = omnic_file.nChannels
    wn = np.arange(wn_min, wn_max+wn_step, wn_step)

    N_X = omnic_file.info['Dim_1']
    N_Y = omnic_file.info['Dim_2']
    X_step = omnic_file.info['OmnicInfo']['Mapping stage X step size']
    Y_step = omnic_file.info['OmnicInfo']['Mapping stage Y step size']

    X = omnic_file.info['positioners']['X']
    Y = omnic_file.info['positioners']['Y']

    data = omnic_file.data

    # generate MAP array
    MAP = np.zeros((N_X,N_Y,2,N_wn))
    MAP[:,:,0,:] = wn

    for i in range(N_X):
        for j in range(N_Y):
            MAP[i,j,1,:] = data[i,j]

    # print information about the map
    print('INFO:')
    print('    file name = ', omnic_file.info['SourceName'][0])
    print('    %d x %d Raman map'%(N_X, N_Y))
    print('    steps = %d um (x), %d um (y)'%(X_step, Y_step))
    print('    from %f to %f, step = %f, # of channels = %d'%(wn_min, wn_max, wn_step, N_wn))
    print('    absolute coordinates: X = [%f - %f]um, Y = [%f - %f]um'%(omnic_file.info['OmnicInfo']['First map location'][0], omnic_file.info['OmnicInfo']['Last map location'][0], omnic_file.info['OmnicInfo']['First map location'][1], omnic_file.info['OmnicInfo']['Last map location'][1]))
    
    return MAP




################################################################
######################## CLASSE SPECTRA ########################
################################################################

# ferma ........

class spectra:

    def __init__(self, S):
        
            if type(S)==str:
                self.info = S[:-4]
                self.S = loadtxt(S)

            elif type(S)==np.ndarray:
                self.S = S
                self.info = 'No uploaded info about this spectra. Save info in this variable'
            elif type(S)==int:
                S = [j for j in os.listdir() if (j[-4:]=='.txt' and j[0:6]!='README')][S]
                self.info = S[:-4]
                self.S = loadtxt(S)
            else:
                print('Type error!')

    def plot(self):
        _raman_labels()
        if self.info!='No uploaded info about this spectra. Save info in this variable':
            plt.title(self.info)
        plt.plot(self.S[0], self.S[1])

    def bkg_subtraction(self):
        self.BKG, self.S_nobkg = bkg_subtraction(self.S)


###################################################################
######################## Initial functions ########################
###################################################################

print('Welcome to pyram: your Raman analysis library!')






###################################################################
######################## ND SEARCH ########################
###################################################################

def ND_cos_sim(S,X):

    if X.shape[0]>1:
        # riempio la matrice X
        X = X.transpose()

        # calcolo la trasposta
        Xt = X.transpose()

        # calcolo la proiezione
        w = np.linalg.multi_dot([X ,np.linalg.inv( np.dot(Xt,X) ), Xt, S[1]])

        # e infine la cosine similarity
        match = 1-distance.cosine(w,S[1])

    if X.shape[0]==1:
        match = 1-distance.cosine(LIB[el],S[1])

    return match


def NDsearch(S, shift, set_min=None, set_max=None, improvement_th = 0.1, fixed_N=None, verbose=False, th=0.01, bin=5):

    ###################################################################################
    ########### PUTTING THE LIBRARY IN AN NP.ARRAY with the computed shifts ###########
    ###################################################################################

    # the nan values are padded with zeros!
    m = len(lib_names)

    # searching the minimum and the maximum wn of the library (to get the # of columns)
    min = lib[lib_names[0]][0].min()
    max = lib[lib_names[0]][0].max()

    for el in lib_names[1:]:
        min_t = lib[el][0].min()
        max_t = lib[el][0].max()

        if min_t<min:
            min = min_t
        if max_t>max:
            max = max_t

    wn = np.arange(min-shift,max+1+shift,1)
    n = len(wn+2*shift)

    # rewriting the library in a np.array
    LIB = np.zeros((m,n))

    for i in range(len(lib_names)):
        min_t = lib[lib_names[i]][0].min()
        max_t = lib[lib_names[i]][0].max()
        
        # computing the shift
        match = 0
        for s in np.arange(-shift,shift+1):
            B_temp = lib[lib_names[i]].copy()
            B_temp[0] += s

            match_new = cos_sim(S,B_temp)                                    
            if match_new>match:
                match = match_new
                out = s

        #adding the shifted spectra to the library
        LIB[i][int(min_t-min+shift)+out:n-int(max-max_t+shift)+out]  = lib[lib_names[i]][1]

    # rescaling the height to 1
    LIB = (LIB.transpose()/LIB.max(axis=1)).transpose()


    ###############################################################################################
    ############# CHANGING LIBRARY OF THE SPECTRA, SPECIFYING FOR THE S WE ARE FACING #############
    ###############################################################################################

    # here we prepare common wn between S and the library

    # ausiliar variables
    S_work = copy.deepcopy(S)
    S_work[1] = S_work[1]/S_work[1].max()

    # changing dimensions of LIB_temp or S_work
    min_1 = np.min(S_work[0])
    min_2 = np.min(wn)
    max_1 = np.max(S_work[0])
    max_2 = np.max(wn)

    # se set_min o set_max sono settati male sistemali
    set_min = np.max([min_1,set_min])
    set_max = np.min([max_1,set_max])

    if set_min==None: #i.e se non è settato un minimo settalo come quello comune
        min = np.max([min_1,min_2])
    else:
        min = set_min
    if set_max==None: #i.e se non è settato un massimo settalo come quello comune
        max = np.min([max_1,max_2])
    else:
        max = set_max

    if min_1<min:
        S_work = S_work[:,int(min-min_1):]
    if min_2<min:
        wn = wn[int(min-min_2):]
        LIB = LIB[:,int(min-min_2):]
    if max_1>max:
        S_work = S_work[:,:int(max-max_1)]
    if max_2>max:
        wn = wn[:int(max-max_2)]
        LIB = LIB[:,:int(max-max_2)]


    #############################################################################################
    ############# LINEAR REGRESSION FOR THE SEARCH OF USEFUL SPECTRA IN THE LIBRARY #############
    #############################################################################################
    
    # binning for the speed up of the linear regression
    LIB_temp = LIB[:,:(LIB.shape[1] // bin)*bin].reshape(LIB.shape[0],LIB.shape[1]//bin,bin).mean(axis=2)
    wn_temp = wn[:(wn.shape[0] // bin)*bin].reshape(wn.shape[0]//bin,bin).mean(axis=1)
    S_temp = S_work[:,:(S_work.shape[1] // bin)*bin].reshape(S_work.shape[0],S_work.shape[1]//bin,bin).mean(axis=2)

    # linear regression
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(LIB_temp.transpose(),S_temp[1])

    # plot the regression
    if verbose==True:
        print('Linear regression to find possible spectra in the library:')
        plt.figure(figsize=(10,5))
        plt.plot(wn_temp,np.dot(LIB_temp.transpose(),reg.coef_))
        plt.plot(S_temp[0],S_temp[1])
        plt.show()

    # number of used spectrums, selection of rely used spectra
    lib_used = []
    for i in range(len(reg.coef_)):
        if reg.coef_[i]>th:
            lib_used.append(i)

    if verbose:
        print('################################################################')
        print('# of used spectra = ',len(lib_used))

    # selected spectra sumup
    sumup = []
    for i in range(len(lib_used)):
        sumup.append([lib_used[i],lib_names[lib_used[i]] , reg.coef_[lib_used[i]]])

    sumup = pd.DataFrame(sumup, columns=['ID','name','regression coefficient'])
    
    if verbose:
        print(sumup)


    #########################################################################################
    ############# GENERATION OF THE REQUIRED SPECTRA IN THE RIGHT CONFIGURATION #############
    #########################################################################################

    # generation of the required pure spectra (also shifted)
    pure = np.take(LIB,list(sumup.ID),axis=0)
    pure.shape

   
    ######################################
    ############# CYCLE ON N #############
    ######################################

    if fixed_N==None:
        N = 0
    else:
        N = fixed_N-1

    improvement = 1
    out = []

    while improvement>improvement_th:
        N += 1

        # genero le N combinazioni tra i pure spectra selezionati
        comb = list(combinations(list(sumup.index),N))

        print('trying N =', N,'; resulting in', len(comb), 'combinations')

        # and finaly let's compute the similarity for each combination
        match = [0]*len(comb)

        for c in enumerate(comb):
            
            if N>1:
                # minimization of the problem
                X = pure.take(c[1],axis=0)

                # store results
                # combination, similarity
                match[c[0]] = [c[1], ND_cos_sim(S_work,X) ]
            
            if N==1:
                tot = pure[c[1][0]]
                match[c[0]] = [c[1], np.dot(S_work[1],tot) / np.sqrt(np.dot(S_work[1],S_work[1])*np.dot(tot,tot)) ]

        # decode results
        for i in range(len(match)):
            temp = [0]*N
            for j in range(N):
                temp[j]= sumup.name[match[i][0][j]]

            match[i].append(match[i][0])
            match[i][0] = temp

        match = pd.DataFrame(match, columns=['combination','match','ID'])


        match.sort_values(by=['match'], inplace=True, ascending=False)
        match.reset_index(inplace=True,drop=True)


        out.append(match)

        if N>1 and fixed_N==None:
            improvement = (out[N-1].match.max()-out[N-2].match.max())/out[N-2].match.max()
        if fixed_N!=None:
            improvement = 0


    if fixed_N==None:  
        final_N = N-1
        out = out[final_N-1]
        
        print('best at N =', final_N)

    else:
        final_N = fixed_N
        out = out[0]
    

    IDs = list(out.iloc[0].ID)
    Names = list(out.iloc[0].combination)

    def f(C):
        tot = np.sum(pure.take(IDs,axis=0).transpose()*C,axis=1)
        #return -np.dot(S_work[1],tot) / np.sqrt(np.dot(S_work[1],S_work[1])*np.dot(tot,tot))
        return np.sum((tot-S_work[1])**2)

    C = minimize(f,np.ones(len(IDs)))['x']

    plt.figure(figsize=(10,5))
    plt.xlabel(r'Raman shift [$cm^{-1}$]')
    plt.ylabel('Intensity [a.u.]')

    plt.plot(S_work[0],S_work[1],label='signal')
    for i in enumerate(IDs):
        plt.plot(wn,pure[i[1]]*C[i[0]], label=Names[i[0]])

    plt.legend()

    return out.head(10).drop('ID',axis=1)