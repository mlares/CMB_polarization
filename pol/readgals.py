def readgals(filename,zmin,zmax,Type):

    import numpy as np

    D = np.genfromtxt(filename,unpack=True,names=True,skip_header=9,delimiter='',dtype=None)
    #D = np.genfromtxt(filename,unpack=True,names=True,delimiter='',dtype=None)


    D = D[D["ba"]<1.] #hay solo un valor choto y es 9.999

    #SELECCION DE GALAXIAS CERCANAS
    D=D[(D['v']/300000.>zmin)&(D['v']/300000.<zmax)]
    #SELECCION DE GALAXIAS 
    #  1    Sa
    #  2    Sab
    #  3    Sb
    #  4    Sbc
    #  5    Sc
    #  6    Scd
    #  7    Sd
    #  8    Sdm
    #  9    Sm, Magellanic Spiral 

    ELLlist=["5","6","7"] #<------ ELIPTICAS
    ALLsplist=["1","2","3","4","5","6","7","8"]#,"9"] #<------ ESPIRALES 

    SAlist=["1","2"] # 1
    SBlist=["3","4"] # 2
    SClist=["5","6"] # 3
    SDlist=["7","8"]#,"9"] #4 

    spty=np.zeros(len(D))

    for i in range(len(D)):

        sp=0

        for j in range(len(SAlist)):
            if((str(D['type'][i])[2]==SAlist[j])):
                  sp=1
        for j in range(len(SBlist)):
            if((str(D['type'][i])[2]==SBlist[j])):
                  sp=2
        for j in range(len(SClist)):
            if((str(D['type'][i])[2]==SClist[j])):
                  sp=3
        for j in range(len(SDlist)):
            if((str(D['type'][i])[2]==SDlist[j])):
                  sp=4
        for j in range(len(ELLlist)):
            if((str(D['type'][i])[2]=='-') & (str(D['type'][i])[3]==ELLlist[j])):
                  sp=-1

        spty[i]=sp

    #........................................................


    #SELECCION DE GALAXIAS por tipo morfológico 
    Dsp=D[spty>0]
    DspSA=D[spty==1]
    DspSB=D[spty==2]
    DspSCD=D[(spty==3) | (spty==4)]
    DspSBCD=D[(spty==2) | (spty==3) | (spty==4)]
    DELL=D[spty==-1]
    if (Type==0):
        Dsel=Dsp
    if (Type==1):
        Dsel=DspSA
    if (Type==2):
        Dsel=DspSB
    if (Type==3):
        Dsel=DspSCD
    if (Type==4):
        Dsel=DspSBCD
    if (Type==-1):
        Dsel=DELL
    if(Type==5):
        Dsel=D

    return(Dsel)

