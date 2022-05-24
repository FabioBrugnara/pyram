#  LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd


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

def bkg_subtraction(S, L_n=41, sigma=150, power=1, p=1000, edge_width=5, edge_weight=2, plot=False, return_bkg=False):

    #type2spectra
    S = type2spectra(S)
    S = interpol(S)

    # smoothing
    S_smooth = S.copy()
    S_smooth[1]=savgol_filter(S[1],L_n,2)

    # vector/matrice Y and X
    Y = S_smooth[1]
    x = S_smooth[0]
    s = sigma/(x[-1]-x[0])
    x = (x-x[0])/(x[-1]-x[0])
    X = np.zeros((len(x),p))
    for i in enumerate(np.linspace(0,1,p)):
        X[:, i[0]] = (np.exp(-(x-i[1])**(2)/(2*s**2)))**power #generative function

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
        plt.plot(XX, S[1,100]/2*(np.exp(-(XX-600)**(2)/(2*sigma**2)))**power,label='Kernel function')
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
def search(S, shift=5, first=10, pre=False):
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
    print(match.head(first))
    return



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