import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class bkg_elimination:
    def __init__(self,noise_window,background_window,threshold=0.05,min_dist=1):
        self.L_n=noise_window
        self.L_b=background_window
        self.threshold=threshold
        self.min_dist=min_dist

    def _moving_average_filter(self,a):
        # the moving average filter with particular boundary conditions
        L=self.L_b
        if L%2==0:
            print('Error, L must be eaven')
        else:
            a=np.convolve(np.array(a),np.ones(L))
            a=np.delete(a,np.concatenate((np.arange(1,L-1,2),np.arange(len(a)-2,len(a)-L,-2))))

            for i in range(len(a)):
                if (L//2+1+i)<L:
                    a[i]=a[i]/(2*i+1)
                elif L//2+(len(a)-i)<L:
                    a[i]=a[i]/((len(a)-i-1)*2+1)
                else:
                    a[i]=a[i]/L
        return a


    def fit(self,signal):
        #Filtro il segnale con un filtro di SG per rimuovere il rumore ad alta frequenza
        s=savgol_filter(signal,self.L_n,2)

        #Calcolo la derivata del segnale e la liscio, questo perchè altrimenti non riesco a lavorarci
        ds=savgol_filter(np.diff(savgol_filter(s,self.L_n,2)),self.L_n,2) #the difference is a vector with one dimension less!!!

        #La derivata del background è poi calcolabile filtrando con una media mobile utilizzando una finestra larca rispetto alla larghezza dei picchi
        s3ds=self._moving_average_filter(self._moving_average_filter(self._moving_average_filter(ds)))
    

        #Così posso trovare la derivata del segnale di cui sono interessato
        dg=ds-s3ds

        #Trovo i centri dei picchi
        temp=[]
        for i in range(1,len(dg)-1):
            if (dg[i-1]>0 and dg[i+1]<0) and dg[i-1]>dg[i] and dg[i]>dg[i+1]:
                temp.append(i)
        peak=[]
        for i in range(1,len(temp)):
            if temp[i]-temp[i-1]==1:
                peak.append(temp[i-1]) 

        #Poi i boundary
        zero=max(dg)*self.threshold
        pl=[]
        pr=[]
        for k in peak:
            left_m=False
            right_m=False
            for i in range(k,k-len(s),-1):
                if (left_m==True and (abs(dg[i])<zero)) or i==0:
                    pl.append(i)
                    break
                if dg[i]<dg[i+1] and left_m==False:
                    left_m=True
            for i in range(k,k+len(s),+1):
                if (right_m==True and (abs(dg[i])<zero)) or i==(len(dg)-1):
                    pr.append(i)
                    break
                if dg[i]>dg[i-1] and right_m==False:
                    right_m=True

        #drop small peaks
        L_max=self.L_n #because L_n is of the order of half of the smallest peak, we could drop everithing smaller

        temp=[]
        for i in enumerate(np.array(pr)-np.array(pl)):
            if i[1]<L_max:
                temp.append(i[0])

        peak=np.delete(np.array(peak),temp)
        pl=np.delete(np.array(pl),temp)
        pr=np.delete(np.array(pr),temp)

        #Unisco i close boundary

        for i in range(len(peak)-1):
            if pl[i+1]-pr[i]<self.min_dist:
                pr[i]=pr[i+1]
                pl[i+1]=pl[i]

        #E infine interpolo il segnale s
        bkg=s.copy()
        for i in range(len(peak)):
            diffline=s3ds[pl[i]:pr[i]]-s3ds[pl[i]:pr[i]].mean()
            m=(s[pl[i]]-s[pr[i]])/(pl[i]-pr[i])
            q=(s[pr[i]]*pl[i]-s[pl[i]]*pr[i])/(pl[i]-pr[i])
            bkg[pl[i]:pr[i]]=m*np.array(range(pl[i],pr[i]))+q+np.cumsum(diffline)
                
        sbkg=self._moving_average_filter(bkg)

        x=signal-sbkg

        self.background=sbkg
        self.signal_no_bkg=x
    
