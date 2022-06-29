#  LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import scipy

from sklearn.linear_model import LinearRegression
from itertools import combinations
from itertools import product

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


txt_alias = 'Run ls() to generate an alias dictionary of the txt files in the current position'
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
    plt.show(block=False)

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
    min = int(S[0][0]) if S[0][0]==0 else int(S[0][0]) +1
    max = int(S[0][-1])
    X = np.arange(min,max+1,1)
    S = np.array([X, np.interp(X,S[0],S[1])])
    return S


#################################################################
######################## BKG SUBTRACTION ########################
#################################################################


from scipy.signal import savgol_filter
from scipy.optimize import linprog

def bkg_subtraction(S, L_n=41, sigma=150, n=1, p=1000, edge_width=5, edge_weight=2, plot=False, return_bkg=False):

    #type2spectra
    S = type2spectra(S)
    S = interpol(S)

    # smoothing
    S_smooth = S.copy()
    S_smooth[1]=savgol_filter(S[1],L_n,n)

    # vector/matrice Y and X
    Y = S_smooth[1]
    x = S_smooth[0]
    s = sigma/(x[-1]-x[0])
    x = (x-x[0])/(x[-1]-x[0])
    X = np.zeros((len(x),p))
    for i in enumerate(np.linspace(0-2*s,1+2*s,p)):
        X[:, i[0]] = (np.exp(-(x-i[1])**(2)/(2*s**2))) #generative function

    # vector C
    H = np.concatenate((np.ones(edge_width)*edge_weight, np.ones(X.shape[0]-2*edge_width), np.ones(edge_width)*edge_weight))
    C = - np.dot(H,X)

    # solving the linear programming problem
    res = linprog(C, A_ub=X, b_ub = Y,bounds = (0, None), method='highs-ds')
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
        plt.show(block=False)

    def bkg_subtraction(self):
        self.BKG, self.S_nobkg = bkg_subtraction(self.S)


###################################################################
######################## Initial functions ########################
###################################################################

print('Welcome to pyram: your Raman analysis library!')






###################################################################
######################## ND SEARCH ########################
###################################################################


def NDsearch(S, shift, th=0.01, improvement_th = 0.1, verbose=False):
    
    ##########################################################
    ########### PUTTING THE LIBRARY IN AN NP.ARRAY ###########
    ##########################################################ù

    global LIB_generation 
    LIB_generation= True
    if LIB_generation:

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

        wn = np.arange(min,max+1,1)
        n = len(wn)

        # rewriting the library in a np.array
        LIB = np.zeros((m,n))

        for i in range(len(lib_names)):
            min_t = lib[lib_names[i]][0].min()
            max_t = lib[lib_names[i]][0].max()
            
            LIB[i][int(min_t-min):n-int(max-max_t)]  = lib[lib_names[i]][1]

        # rescaling the height to 1
        LIB = (LIB.transpose()/LIB.max(axis=1)).transpose()

        LIB_generation = False

    ###########################################################################################################
    ############# GENERATING A LIBRARY OF THE SHIFTED SPECTRA, SPECIFYING FOR THE S WE ARE FACING #############
    ###########################################################################################################

    # here we prepare common wn between S and the library

    # ausiliar variables
    LIB_temp = copy.deepcopy(LIB)
    S_temp = copy.deepcopy(S)
    S_temp[1] = S_temp[1]/S_temp[1].max()
    wn_temp = copy.deepcopy(wn)

    # changing dimensions of LIB_temp or S_temp
    min_1 = np.min(S_temp[0])
    min_2 = np.min(wn_temp)

    if min_1<min_2:
        S_temp = S_temp[int(min_2-min_1):]
    if min_2<min_1:
        wn_temp = wn_temp[int(min_1-min_2):]
        LIB_temp = LIB_temp[:,int(min_1-min_2):]

    max_1 = np.max(S_temp[0])
    max_2 = np.max(wn_temp)

    if max_1>max_2:
        S_temp = S_temp[:,:int(max_2-max_1)]
    if max_2>max_1:
        wn_temp = wn_temp[:int(max_1-max_2)]
        LIB_temp = LIB_temp[:,:int(max_1-max_2)]

    # generate shifted spectra in LIB_temp
    shift_vec = np.arange(-shift,shift+1)

    LIB_temp_shifted = np.zeros((LIB_temp.shape[0]*len(shift_vec),LIB_temp.shape[1]))

    for i in range(len(shift_vec)):
        if shift_vec[i]>0:
            LIB_temp_shifted[i*LIB_temp.shape[0]:(i+1)*LIB_temp.shape[0],int(shift_vec[i]):] = LIB_temp[:,:-int(shift_vec[i])]
        if shift_vec[i]<0:
            LIB_temp_shifted[i*LIB_temp.shape[0]:(i+1)*LIB_temp.shape[0],:int(shift_vec[i])] = LIB_temp[:,-int(shift_vec[i]):]
        if shift_vec[i]==0:
            LIB_temp_shifted[i*LIB_temp.shape[0]:(i+1)*LIB_temp.shape[0],:] = LIB_temp[:,:]

    LIB_temp = LIB_temp_shifted
    del LIB_temp_shifted

    #############################################################################################
    ############# LINEAR REGRESSION FOR THE SEARCH OF USEFUL SPECTRA IN THE LIBRARY #############
    #############################################################################################
    
    # linear regression
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(LIB_temp.transpose(),S_temp[1])


    if verbose:
        # plot the regression
        plt.figure(figsize=(10,5))
        plt.plot(wn_temp,np.dot(LIB_temp.transpose(),reg.coef_))
        plt.plot(S_temp[0],S_temp[1])
        plt.show()

    # delete (big) LIB_temp variable
    del LIB_temp

    # number of used spectrums used, selection of rely used spectra
    count=0
    lib_used = []
    for i in range(len(reg.coef_)):
        if reg.coef_[i]>th:
            count+=1
            lib_used.append(i)

    if verbose:
        print('################################################################')
        print('# of used spectra = ',count)

    # selected spectra sumup
    sumup = []
    for i in range(len(lib_used)):
        sumup.append([lib_names[lib_used[i] - m*(lib_used[i]//m)] , shift_vec[lib_used[i]//m], reg.coef_[lib_used[i]]])

    sumup = pd.DataFrame(sumup, columns=['name','shift','regression coefficient'])

    if verbose:    
        print(sumup)


    #########################################################################################
    ############# GENERATION OF THE REQUIRED SPECTRA IN THE RIGHT CONFIGURATION #############
    #########################################################################################

    # generation of the required pure spectra (also shifted)
    pure = [0] * len(sumup.index)
    for i in sumup.index:
        pure[i] = copy.deepcopy(lib[sumup.name[i]])
        pure[i][0] = pure[i][0]+sumup['shift'][i]
        pure[i][1] = pure[i][1]/pure[i][1].max()


    # per semplicità restringiamo subito tutto allo stesse common wn

        # calcolo min e max della combinazione

    min = S[0].min()
    max = S[0].max()

    for el in sumup.index:
        min_pure = pure[el][0].min()
        max_pure = pure[el][0].max()

        if min_pure>min:
            min = min_pure
        if max_pure<max:
            max = max_pure
        # e ora restringiamo

            # prima S in S_temp
    S_temp = copy.deepcopy(S)
    S_temp[1] = S_temp[1]/S_temp[1].max()

    if S_temp[0].min()<min:
        S_temp = S_temp[:,int(min-S_temp[0].min()):]
    if S_temp[0].max()>max:
        S_temp = S_temp[:,:-int(S_temp[0].max()-max)]

            # poi i pure spectra
    for el in sumup.index:
        if pure[el][0].min()<min:
            pure[el] = pure[el][:,int(min-pure[el][0].min()):]
        if pure[el][0].max()>max:
            pure[el] = pure[el][:,:-int(pure[el][0].max()-max)]

    #############################################################
    ############# FINAL FIT USING COSINE SIMILARITY #############
    #############################################################

    N = 0
    improvement = 1
    out = []

    while improvement>improvement_th:
        N += 1

        if N == 1:
            print('trying N =', N)
            match = search(S, shift, verbose=False)
            match.drop(columns=['alias'], inplace=True)

        if N>1:

            # genero le N combinazioni tra i pure spectra selezionati
            comb = list(combinations(list(sumup.index),N))
            # eliminare combinazioni dello stesso spettro shiftato??????


            print('trying N =', N,'; resulting in', len(comb), 'combinations')

            # and finaly let's compute the similarity for each combination
            match = [0]*len(comb)

            for c in enumerate(comb):
                
                # minimization of the problem
                def fun(intensity):
                    intensity = np.concatenate(([1],intensity))
                    tot = np.zeros(len(S_temp[1]))

                    for i in range(N):
                        tot += pure[c[1][i]][1] * intensity[i]
                    return -np.dot(S_temp[1],tot) / np.sqrt(np.dot(S_temp[1],S_temp[1])*np.dot(tot,tot))

                X = scipy.optimize.minimize(fun, [1]*(N-1), method='Nelder-Mead', bounds=[(0,100)]*(N-1))
                if X.success==False:
                    print('error in the optimization!!!')

                # store results
                # combination, intensity, similarity
                match[c[0]] = [c[1], X.fun ]
                

            for i in range(len(match)):
                temp = [0]*N
                for j in range(N):
                    temp[j]= sumup.name[match[i][0][j]]

                match[i][0] = temp
            match = pd.DataFrame(match, columns=['name','match'])

            match['match'] = -match['match']
            match.sort_values(by=['match'], inplace=True, ascending=False)
            match.reset_index(inplace=True, drop=True)


        out.append(match)
        if N>1:
            improvement = (out[N-1].match.max()-out[N-2].match.max())/out[N-2].match.max()
        

    print('########################################################')
    print('best at N =', N-1)
    print(out[N-2].head(10))
    print('########################################################')
    return out[N-2]
