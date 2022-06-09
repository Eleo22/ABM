import numpy as np, pandas as pd, matplotlib.pyplot as plt
#from os.path import join
import random
import networkx as nx
from pynverse import inversefunc
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
import time
from scipy.stats import nbinom
import scipy.special as sps  
from sklearn.utils import shuffle

import json
with open("../params.dat") as file:
    inputf = json.load(file)




random.seed(inputf["jranini"])

################

Nindiv=inputf["Nindiv"] #75000
Nsim=inputf["Nsim"]
Nt0_inf=int(Nindiv*inputf["fn0"])
#int(Nindiv*0.0011)
R=5

Ntestperw=[0] #[0,round(500000/67000000*Nindiv),round(1000000/67000000*Nindiv),round(1500000/67000000*Nindiv)]

################ Building the networks
dfAdj=pd.read_csv("../../Maestro_house/household_dem3.csv",sep=";")
#dfAdj=pd.read_csv("../../../../../../Vaccination/codes/household_dem3.csv",sep=";")
Xmattemp=np.array(dfAdj)
Nhouse=50000
Xmat=np.empty((Nhouse,9))
ih=[i for i in range(Nhouse)]
random.shuffle(ih)
for j in range(0,Nhouse):
    ihouse=ih[j]
    for iage in range(0,9):
            Xmat[j,iage]=Xmattemp[ihouse,iage]

ic_indiv=-1    
NageMoreno=np.zeros(9)

# define agents and assign them to a house
inte=[i for i in range(Nindiv)]
random.shuffle(inte)
idagent_temp=-1
Ageindiv=np.zeros(Nindiv)
cluster_dict = dict()
Agent_house=np.zeros(Nindiv)

for ihouse in range(0,Nhouse):
    list_h=[]
    for iage in range(0,9):
        
        ic_indiv+=Xmat[ihouse,iage]
        NageMoreno[iage]+=Xmat[ihouse,iage]
        
        Nag=int(Xmat[ihouse,iage])
        for j in range(0,Nag):
            idagent_temp+=1
            if idagent_temp<Nindiv:  
                idagent=inte[idagent_temp]
                Ageindiv[idagent]=random.uniform(1, 9)+10*iage
                Agent_house[idagent]=(ihouse)
              #  print("idagent=",idagent,"of age=",Ageindiv[idagent],"is in house",ihouse)
                list_h.append(idagent)  
                cluster_dict[ihouse]=list_h 
                nbhouse=ihouse

#print(cluster_dict)
nbhouse=nbhouse+1
print("there are", nbhouse, "houses in total and",Nindiv,"agents")

Nchildren=0
Neldery=0
Nteen=0
for i in range(0,Nindiv):
    if Ageindiv[i]<=9:
        Nchildren+=1
    elif Ageindiv[i]>9 and Ageindiv[i]<=19:
        Nteen+=1
    elif Ageindiv[i]>49: 
        Neldery+=1
        
Nadult=Nindiv-Nchildren-Neldery-Nteen
print("Nindiv=",Nindiv,"Nadult=",Nadult,"Nchildren=",Nchildren,"Nteen=",Nteen,"Neldery=",Neldery)

# arrange the IDs of agents as function of their occupation
Id_children=np.zeros(Nchildren)
Id_teen=np.zeros(Nteen)
Id_childandteen=np.zeros(Nteen+Nchildren)
Id_adult=np.zeros(Nadult)
Id_eldery=np.zeros(Neldery)
Agent_cat=np.zeros(Nindiv)
ic=-1
ia=-1
ie=-1
it=-1
icandt=-1
for i in range(0,Nindiv):
    if Ageindiv[i]<=9 :
        ic+=1
        Id_children[ic]=i 
    elif Ageindiv[i]>9 and Ageindiv[i]<=19:
        it+=1
        Id_teen[it]=i
    elif Ageindiv[i]>49:
        ie+=1
        Id_eldery[ie]=i
        Agent_cat[i]=2    
    else:
        ia+=1
        Id_adult[ia]=i
        Agent_cat[i]=1 
        
    if Ageindiv[i]<=19:
        icandt+=1
        Id_childandteen[icandt]=i
        Agent_cat[i]=0


#2c define workplace interactions between agents using Watts-Strogatz small-world networks 
#2c.1 define the networks:
Agenttake=np.zeros(Nindiv)
clusterwork_dict = dict()
Agent_work=np.zeros(Nindiv)

       
Nfrac_sh=round(0.2*Nchildren) # number of adult who interact daily with childrens at schools 0.2 by default
Nfrac_teen=round(0.2*Nteen) 
Nfrac_el=round(0.2*Neldery)   # number of adult who interact daily with eldery  0.2 by default

print("Nchildren=",Nchildren,"frac_sh",Nfrac_sh,"Nteen=",Nteen,"Neldery=",Neldery,"frac_el",Nfrac_el,"Nadult",Nadult)    


G_children = nx.watts_strogatz_graph(n=Nchildren+Nfrac_sh, k=10, p=0.1) #mean daily interac=k
G_Teen = nx.watts_strogatz_graph(n=Nteen+Nfrac_teen, k=10, p=0.1) 
G_adult = nx.watts_strogatz_graph(n=Nadult, k=7, p=0.1)    
G_eldery = nx.watts_strogatz_graph(n=Neldery+Nfrac_el, k=7, p=0.1)     

Adj=nx.to_numpy_array(G_children)
Adjteen=nx.to_numpy_array(G_Teen)
Adjel=nx.to_numpy_array(G_eldery)
Adjw=nx.to_numpy_array(G_adult)
print("start filling net")
 

#2c.2 assign agent to the networks - for schools_childrens:
Agentworktake=np.zeros(Nadult)
cluster_schools_dict = dict()
Agent_school=np.zeros(Nindiv)-1
node=np.zeros(Nchildren+Nfrac_sh)
idnode=np.zeros(Nchildren+Nfrac_sh)
ic1=-1

for i in range(0,Nchildren+Nfrac_sh):
    
    list_s=[]
    inter=np.argwhere(Adj[:,i] ==1)
   
    
    if i>Nchildren-1 and node[i]!=i:
        node[i]=i
        ic1+=1
        i1=int(Id_adult[ic1])
        Agentworktake[ic1]=1
 
        idnode[i]=i1
        
    elif i>Nchildren-1 and node[i]==i:
        i1=int(idnode[i])
        Agentworktake[ic1]=1

    else:    
        i1=int((Id_children[i]))
            
    for j in inter:
   
       
        if j>Nchildren-1 and node[j]!=j: 
            ic1+=1
            i2=int(Id_adult[ic1])
            Agentworktake[ic1]=1
   
            node[j]=j 
            idnode[j]=i2
            
        elif j>Nchildren-1 and node[j]==j:  
            i2=int(idnode[j]) 
     
        else:    
            i2=int(Id_children[j])
    
        list_s.append(i2)
        
        
    cluster_schools_dict[i]=[i1]+list_s
    Agent_school[i1]=(i)


#2c.2 assign agent to the networks - for schools for teens:
### do not re-initialized Agentworktake=np.zeros(Nadult)
cluster_teen_dict = dict()
Agent_teen=np.zeros(Nindiv)-1
node=np.zeros(Nteen+Nfrac_teen)
idnode=np.zeros(Nteen+Nfrac_teen)

##### do no reinitialized !!! ic1=-1

for i in range(0,Nteen+Nfrac_teen):
    
    list_t=[]
    inter=np.argwhere(Adjteen[:,i] ==1)
   # print("node i=",i)
    
    if i>Nteen-1 and node[i]!=i:
        node[i]=i
        ic1+=1
        i1=int(Id_adult[ic1])
        Agentworktake[ic1]=1
    #    print("nodei=",i,">",Nchildren,"so we add up adults agents nb",i1)
        idnode[i]=i1
        
    elif i>Nteen-1 and node[i]==i:
        i1=int(idnode[i])
        Agentworktake[ic1]=1
    #    print("nodei=",i,">",Nchildren,"so we take agents nb",i1)
    else:    
        i1=int((Id_teen[i]))
            
    for j in inter:
    #    print("node i",i,"is in contact with node j",j)
       
        if j>Nteen-1 and node[j]!=j: 
            ic1+=1
            i2=int(Id_adult[ic1])
            Agentworktake[ic1]=1
     #       print("within node i =",i,"nodej=",j,">",Nchildren,"so we add up adults agents nb",i2)
            node[j]=j 
            idnode[j]=i2
            
        elif j>Nteen-1 and node[j]==j:  
            i2=int(idnode[j]) 
      #      print("within node i=",i,"nodej=",j,">",Nchildren,"so we take agent",i2)
        else:    
            i2=int(Id_teen[j])
      #  print("agent",i1,"is in contact with agent",i2,i)
             
        list_t.append(i2)
        
        
    cluster_teen_dict[i]=[i1]+list_t
    Agent_teen[i1]=(i)


#2c.2 assign agent to the networks - eldery:
#### Agentworktake=np.zeros(Nadult) was defined for the school and should not be redefined
cluster_eldery_dict = dict()
Agent_eldery=np.zeros(Nindiv)-1
node=np.zeros(Neldery+Nfrac_el)
idnode=np.zeros(Neldery+Nfrac_el)
### was previously defined and should not be used ic1=-1

for i in range(0,Neldery+Nfrac_el):
    
    list_e=[]
    inter=np.argwhere(Adjel[:,i] ==1)
  #  print("node i=",i)
    
    if i>Neldery-1 and node[i]!=i:
        node[i]=i
        ic1+=1
        i1=int(Id_adult[ic1])
        Agentworktake[ic1]=1
  #      print("nodei=",i,">",Nchildren,"so we add up adults agents nb",i1)
        idnode[i]=i1
        
    elif i>Neldery-1 and node[i]==i:
        i1=int(idnode[i])
        Agentworktake[ic1]=1
   #     print("nodei=",i,">",Nchildren,"so we take agents nb",i1)
    else:    
        i1=int((Id_eldery[i]))
            
    for j in inter:
  #      print("node i",i,"is in contact with node j",j)
       
        if j>Neldery-1 and node[j]!=j: 
            ic1+=1
            i2=int(Id_adult[ic1])
            Agentworktake[ic1]=1
   #         print("within node i =",i,"nodej=",j,">",Nchildren,"so we add up adults agents nb",i2)
            node[j]=j 
            idnode[j]=i2
            
        elif j>Neldery-1 and node[j]==j:  
            i2=int(idnode[j]) #int(Id_adult[j])
            #Agentworktake[ic1]=1
  #          print("within node i=",i,"nodej=",j,">",Nchildren,"so we take agent",i2)
        else:    
            i2=int(Id_eldery[j])
     #   print("agent",i1,"is in contact with agent",i2,i)
             
        list_e.append(i2)
        
        
    cluster_eldery_dict[i]=[i1]+list_e
    Agent_eldery[i1]=(i)

#2c.3 assign agent to the networks - workplace:
cluster_work_dict = dict()
Agent_work=np.zeros(Nindiv)-1

for i in range(0,Nadult):   
    list_w=[]
    inter=np.argwhere(Adjw[:,i] ==1)
    i1=int((Id_adult[i]))
            
    for j in inter: 
        i2=int(Id_adult[j])
        list_w.append(i2)
        
        
    cluster_work_dict[i]=[i1]+list_w
    Agent_work[i1]=(i)


######################3 compute Nperagebin:
     
Nperage=np.zeros(9)

for i in range(0,Nindiv):
    binage=int(Ageindiv[i]/10.)
    if binage>8:
        binage=8
    Nperage[binage]+=1



output8=open('simustatr0.csv', 'w')
output8.write('iage,Niage,Nindiv,Nsim'+'\n')        
for i in range(0,9):
    output8.write(str(i)+ ", " + str(Nperage[i])+", " + str(Nindiv) + ", " + str(Nsim)+"\n")    
output8.close()
    
  
    
##################### start simulations:
########################################################################################################
####
########################################################################################################

    

#3 define simulation initial param
start_time = time.time() 
from scipy.stats import gamma
mean_rand_int=[2.2,2,2.] 
std_rand_int=[2,2,2]  #for uk[2,4,3]

#### params of the simulation    
tmax=inputf["tmax"]
t=np.arange(tmax)
test_insensitive_period=3 # 3 days after infection agent can be tested positif


##### Ptrans parameters:
#[0-9,10-19,20-29,...80+]
fraction_asymp=[0.456,0.412,0.370,0.332,0.296,0.265,0.238,0.214,0.192]
##########R=1.5*2 #5.18
Susceptibility=[0.35,0.69,1.03,1.03,1.03,1.03,1.27,1.52,1.52] # suseptibility to infection/age bins
Aasym=0.33 # infectious rate factor for infected asymptomatic agent 
Amild=0.72 # infectious rate factor for infected agent with mild symptomes
A=[Aasym,Amild]
Bhome=2
Bother=1
Bran=1

mean_infectious=5.5
std_infectious=2.14
#gamma.pdf(iday,a,Datet[agent_i],scale)
a=(mean_infectious/std_infectious)**2
scale=mean_infectious/a       


avdailyinteract=3.7-R**0.3 #2.38 #((7/2+5)*(Nchildren+Nteen)+(6/2+4)*Nadult+(3/2+4)*Neldery)/(Nindiv)+2.2
###### params for PDF(t) symptoms apparition 
mean_time_to_symptoms=5.42
sd_time_to_symptoms=2.7
shape_s=(mean_time_to_symptoms/sd_time_to_symptoms)**2
scale_s=mean_time_to_symptoms/shape_s


##### params for PDF of time recovery 
mean_time_to_recover=12 #mean time to recover if hospital isn't required Yang et al. 2020 or 15 for asympto?
sd_time_to_recover=5
shape_r=(mean_time_to_recover/sd_time_to_recover)**2
scale_r=mean_time_to_recover/shape_r
frac_hosp=[0.001,0.006,0.015,0.069,0.219,0.279,0.370,0.391,0.379]


Agent_symp=np.zeros(Nindiv)
Agent_rancont=np.zeros(Nindiv)
Agent_timetosymptoms=np.zeros(Nindiv)
Agent_timerecovery=np.zeros(Nindiv)
Agent_binage=np.zeros(Nindiv)

#nremain=Nindiv
for i in range(0,Nindiv):
    ran=random.uniform(0, 1)
    binage=int(Ageindiv[i]/10.)
    if binage>8:
        binage=8
    Agent_binage[i]=binage
    
    t_to_symptoms=np.random.gamma(shape_s,scale_s,size=1)
    Agent_timetosymptoms[i]=t_to_symptoms
    
######### set up if agent will have symptomes and t_recovery    
    # initial time recovery which get changed if agent go to hospital
    trec=np.random.gamma(shape_r,scale_r,size=1)
    Agent_timerecovery[i]=trec  
    
    if(ran>fraction_asymp[binage]):
        isymp=1
        Agent_symp[i]=(isymp)
    # here we add a fraction of agents who go to hospital and hence
   # can not transmit after 5-6 days (ideally from Bernoulli distribution but don't know p!)
        ran2=random.uniform(0, 1)
        if ran<frac_hosp[binage]:
            Agent_timerecovery[i]=5 #5 or 6
        
###### set the nb of random contacts an agent make per day    
    #if nremain>=0:
    icat=int(Agent_cat[i])
    p=mean_rand_int[icat]/std_rand_int[icat]**2
    n=mean_rand_int[icat]*p/(1.-p)
    Nran=(np.random.negative_binomial(n, p))
    Agent_rancont[i]=(Nran)
        #nremain-=Nran

###### parameters for school in quarantines:
#quarantine_school_cluster=np.zeros(len(cluster_schools_dict))
#quarantine_teen_cluster=np.zeros(len(cluster_teen_dict))
#Dateqs=np.zeros(len(cluster_schools_dict))
#Dateqt=np.zeros(len(cluster_teen_dict))
tquarantine=7
#Agent_quarantine_s=np.zeros(Nindiv)     
#countReff=np.zeros(Nindiv)
#### tests properties:
#tpositif=3 #  a test can detect infection 3 days after the infection 
tpositif_inf=3
tpositif_max=18

#frac_vacci=[0.,0.3*Nperage[1],0.78*Nperage[2],0.78*Nperage[3],0.78*Nperage[4],0.78*Nperage[5],0.90*Nperage[6],0.9*Nperage[7],0.9*Nperage[8]]

#frac_vacci=[0.,0.7*Nperage[1],0.9*Nperage[2],0.9*Nperage[3],0.9*Nperage[4],0.9*Nperage[5],0.90*Nperage[6],0.9*Nperage[7],0.9*Nperage[8]]

#frac_vacci=[0.2*Nperage[0],0.2*Nperage[1],0.2*Nperage[2],0.2*Nperage[3],0.2*Nperage[4],0.2*Nperage[5],0.2*Nperage[6],0.2*Nperage[7],0.2*Nperage[8]]

#frac_vacci=[0.9*Nperage[0],0.9*Nperage[1],0.9*Nperage[2],0.9*Nperage[3],0.9*Nperage[4],0.9*Nperage[5],0.9*Nperage[6],0.9*Nperage[7],0.9*Nperage[8]]
#frac_vacci=[0.75*Nperage[0],0.75*Nperage[1],0.75*Nperage[2],0.75*Nperage[3],0.75*Nperage[4],0.75*Nperage[5],0.75*Nperage[6],0.75*Nperage[7],0.75*Nperage[8]]

#frac_vacci=[0.8*Nperage[0],0.8*Nperage[1],0.8*Nperage[2],0.8*Nperage[3],0.8*Nperage[4],0.8*Nperage[5],0.8*Nperage[6],0.8*Nperage[7],0.8*Nperage[8]]


ptvac=inputf["ptvac"] #0.5
pivac=inputf["pivac"]


#######################################################################    
##############################run simulations##########################
#######################################################################
#######################################################################
print("R=",R)
fstep=inputf["fstep"]
for ivac in range (5,10):
    if ivac==5:
        ivac=0
        frac_vacci=[0]*9
    else:
        
        frac_vacci=[ivac*fstep*Nperage[0],ivac*fstep*Nperage[1],ivac*fstep*Nperage[2],ivac*fstep*Nperage[3],ivac*fstep*Nperage[4],ivac*fstep*Nperage[5],ivac*fstep*Nperage[6],ivac*fstep*Nperage[7],ivac*fstep*Nperage[8]]
    
    frac_vacc=np.array(frac_vacci)
    Nvac=np.sum(frac_vacc)
    print("we have",Nvac,"vaccin")
    print(frac_vacc)
    
####################### make r-->0
## start with random assignment
    Ar=10.
    il=0
    Arold=Ar
    while il<30: #np.abs(Ar)>0.392:
        il+=1
        Immuned=np.zeros(Nindiv)
        cc=np.zeros(9)
        ir=[i for i in range(Nindiv)]
        random.Random(il).shuffle(ir)
        nvacc=0
        j=0
        while Nvac>nvacc:
            j+=1
            i=ir[j]
            binage=int(Ageindiv[i]/10.)
            if binage>8:
                binage=8
            if frac_vacc[binage]>cc[binage] and Immuned[i]==0:
                cc[binage]+=1
                Immuned[i]=1
                nvacc+=1
                
        ###### compute r
        ncontact=np.zeros(Nindiv)
        e00=np.zeros(Nindiv)
        e11=np.zeros(Nindiv)
        e10=np.zeros(Nindiv)
        e01=np.zeros(Nindiv)
        for i in range(0,Nindiv):
            ihouse=Agent_house[i]
            for  key in cluster_dict[ihouse]:
                if i !=key:
                    ncontact[i]+=1
                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                    e11[i]+=1
                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                    e00[i]+=1
                elif Immuned[i]==1 and Immuned[key]==0 :
                    e10[i]+=1
               
            if Agent_school[i]>-1:
                ischool=int(Agent_school[i])
                for key in cluster_schools_dict[ischool]:
                    if i !=key:
                        ncontact[i]+=1
                    if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                        e11[i]+=1
                    elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                        e00[i]+=1
                    elif Immuned[i]==1 and Immuned[key]==0 :
                        e10[i]+=1
                            
                        
            if Agent_teen[i]>-1:
                iteen=int(Agent_teen[i])
                for key in cluster_teen_dict[iteen]:
                    if i !=key:
                        ncontact[i]+=1
                    if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                        e11[i]+=1
                    elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                        e00[i]+=1
                    elif Immuned[i]==1 and Immuned[key]==0 :
                        e10[i]+=1
                        
            if Agent_eldery[i]>-1:
                ieldery=int(Agent_eldery[i])
                for key in cluster_eldery_dict[ieldery]:
                    if i !=key:
                        ncontact[i]+=1
                    if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                        e11[i]+=1
                    elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                        e00[i]+=1
                    elif Immuned[i]==1 and Immuned[key]==0 :
                        e10[i]+=1
                     #   print("el",i,key)

            if Agent_work[i]>-1:
                iwork=int(Agent_work[i])
                for key in cluster_work_dict[iwork]:
                    if i !=key:
                        ncontact[i]+=1
                    if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                        e11[i]+=1
                    elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                        e00[i]+=1
                    elif Immuned[i]==1 and Immuned[key]==0 :
                        e10[i]+=1
            
            
                    
        e00g=np.sum(e00)/np.sum(ncontact)
        e10g=(np.sum(e10)/np.sum(ncontact))
        e01g=e10g
        e11g=np.sum(e11)/np.sum(ncontact)
        e=np.matrix( [[e00g,e01g],[e10g,e11g] ])
        Ar=(np.trace(e)-np.sum(np.dot(e,e)))/(1.-np.sum(np.dot(e,e)))
        
        if Arold*0.993>Ar and il >1:
            print("break", Arold,Ar,Arold*0.993 ,il)
            break
        
        if Ar<Arold and il<2:
            Arold=Ar
       
        print("after rd attibution",il,Arold,Ar,Arold*0.99)
   
    ###### add global switch
    #print("mean,med,std e00",np.mean(e00),np.median(e00),np.std(e00))
    #print("mean,med,std e10",np.mean(e10),np.median(e00),np.std(e00))
    subset=[0]
    e00bis=[i for i in e00 if i not in subset]
    #print("mean/med/std e00bis=",np.mean(e00bis),np.median(e00bis),np.std(e00bis))
    e10bis=[i for i in e10 if i not in subset]
    #print("mean e10bis, e10=",np.mean(e10bis),np.mean(e10))
    
    ###  method global hit and miss from mean nb of edges for R--->-1
    
    Arr=Ar
    Arg=Ar #10
    il=0
    Arold=-1
    ial=0
    bach=100
    print("we start from Ar=",Ar)
    ir=[i for i in range(Nindiv)]
    random.shuffle(ir)
    leftover=list(range(Nindiv))

    while np.abs(Ar)>1: #not a problem as the loop break when no replacement is found
        ial+=1
        j=np.argmax(e11)
        ok=0
        Immuned[j]=0
        binage=int(Ageindiv[j]/10.)
        if binage>8:
            binage=8
        random.shuffle(leftover)

        for j2 in leftover:
            binage2=int(Ageindiv[j2]/10.)
            if binage2>8:
                binage2=8
            if j2!=j and binage2==binage and Immuned[j2]==0 and (e10[j2]>e00[j2] or e00[j2]>np.mean(e00bis) or e10[j2]>np.mean(e10bis)):
                Immuned[j2]=1
                leftover.remove(j)
                ok=1
                e11[j]=0
                e00[j2]=0
                break


        if ok !=1:
            Immuned[j]=1
     #       print("we can't find a replacement problem")
            break
                    
                    
      
        
        
        if ial%int(bach)==0:
            ###### compute r until it descreases
            ncontact=np.zeros(Nindiv)
            e00=np.zeros(Nindiv)
            e11=np.zeros(Nindiv)
            e10=np.zeros(Nindiv)
            e01=np.zeros(Nindiv)
            for i in range(0,Nindiv):
                ihouse=Agent_house[i]
                for  key in cluster_dict[ihouse]:
                    if i !=key:
                        ncontact[i]+=1
                    if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                        e11[i]+=1
                    elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                        e00[i]+=1
                    elif Immuned[i]==1 and Immuned[key]==0 :
                        e10[i]+=1

                if Agent_school[i]>-1:
                    ischool=int(Agent_school[i])
                    for key in cluster_schools_dict[ischool]:
                        if i !=key:
                            ncontact[i]+=1
                        if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                            e11[i]+=1
                        elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                            e00[i]+=1
                        elif Immuned[i]==1 and Immuned[key]==0 :
                            e10[i]+=1


                if Agent_teen[i]>-1:
                    iteen=int(Agent_teen[i])
                    for key in cluster_teen_dict[iteen]:
                        if i !=key:
                            ncontact[i]+=1
                        if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                            e11[i]+=1
                        elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                            e00[i]+=1
                        elif Immuned[i]==1 and Immuned[key]==0 :
                            e10[i]+=1

                if Agent_eldery[i]>-1:
                    ieldery=int(Agent_eldery[i])
                    for key in cluster_eldery_dict[ieldery]:
                        if i !=key:
                            ncontact[i]+=1
                        if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                            e11[i]+=1
                        elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                            e00[i]+=1
                        elif Immuned[i]==1 and Immuned[key]==0 :
                            e10[i]+=1
                                 #   print("el",i,key)

                if Agent_work[i]>-1:
                    iwork=int(Agent_work[i])
                    for key in cluster_work_dict[iwork]:
                        if i !=key:
                            ncontact[i]+=1
                        if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                            e11[i]+=1
                        elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                            e00[i]+=1
                        elif Immuned[i]==1 and Immuned[key]==0 :
                            e10[i]+=1



            e00g=np.sum(e00)/np.sum(ncontact)
            e10g=(np.sum(e10)/np.sum(ncontact))
            e01g=e10g
            e11g=np.sum(e11)/np.sum(ncontact)
            e=np.matrix( [[e00g,e01g],[e10g,e11g] ])
            Ar=(np.trace(e)-np.sum(np.dot(e,e)))/(1.-np.sum(np.dot(e,e)))
      #      print(" check ial=",ial, Arg,"fracind=",ial/Nindiv,Arg/Arr,np.max(e11),np.mean(e11),np.max(e00),np.mean(e00))
            
    #print("now we start the systematic loop from",Arg,ial,np.max(e11),np.mean(e11),np.max(e00),np.mean(e00))
    #### reloop in a systematic way slower :######################
    #
    #
    ### we do the same for e00
      ######
    ial=0
    leftover=list(range(Nindiv))
    #print("max,mean,std e11=",np.max(e11),np.mean(e11),np.std(e11))
    print("we remove vaccins from np.max(e00)=",np.max(e00))
    while Ar>1: #np.max(e00)>np.mean(e00)+np.std(e00) or ial<Nindiv-1: #np.max(e11)>np.mean(e11):
        j=np.argmax(e00) # taking the node with the max number of e11
        ial+=1
        ok=0
        if np.abs(Ar)<0.09:
            break
           
        if Immuned[j]==0 :
            Immuned[j]=1
            binage=int(Ageindiv[j]/10.)
            if binage>8:
                binage=8
            random.shuffle(leftover)
                
            for j2 in leftover:
                binage2=int(Ageindiv[j2]/10.)
                if binage2>8:
                    binage2=8
                if j2!=j and binage2==binage and Immuned[j2]==1 and e11[j2]>e10[j2] or e11[j2]>np.mean(e11)+np.std(e11):
                    Immuned[j2]=0
                        
                    ###### compute r until it descreases
                    ncontact=np.zeros(Nindiv)
                    e00=np.zeros(Nindiv)
                    e11=np.zeros(Nindiv)
                    e10=np.zeros(Nindiv)
                    e01=np.zeros(Nindiv)
                    for i in range(0,Nindiv):
                        ihouse=Agent_house[i]
                        for  key in cluster_dict[ihouse]:
                            if i !=key:
                                ncontact[i]+=1
                            if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                e11[i]+=1
                            elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                e00[i]+=1
                            elif Immuned[i]==1 and Immuned[key]==0 :
                                e10[i]+=1

                        if Agent_school[i]>-1:
                            ischool=int(Agent_school[i])
                            for key in cluster_schools_dict[ischool]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1


                        if Agent_teen[i]>-1:
                            iteen=int(Agent_teen[i])
                            for key in cluster_teen_dict[iteen]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1

                        if Agent_eldery[i]>-1:
                            ieldery=int(Agent_eldery[i])
                            for key in cluster_eldery_dict[ieldery]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1
                                     #   print("el",i,key)

                        if Agent_work[i]>-1:
                            iwork=int(Agent_work[i])
                            for key in cluster_work_dict[iwork]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1



                    e00g=np.sum(e00)/np.sum(ncontact)
                    e10g=(np.sum(e10)/np.sum(ncontact))
                    e01g=e10g
                    e11g=np.sum(e11)/np.sum(ncontact)
                    e=np.matrix( [[e00g,e01g],[e10g,e11g] ])
                    Arg=(np.trace(e)-np.sum(np.dot(e,e)))/(1.-np.sum(np.dot(e,e)))
                    # print(il,Arg)
                    if Arg<Ar:
                          #  print(j,"is replaced by",j2, Arg,Ar)
                        Ar=Arg
                        ok=1
                        leftover.remove(j)
                           # leftover.remove(j2)
                        break
                    else:
                        Immuned[j2]=1
            if ok!=1:
                Immuned[j]=0
                print(j,"we didn't found a better node",j2, Arg,Ar)
                break
                        
          
    print("we end with Ar=",Ar,"with e11 e00 max=",np.max(e11),np.max(e00),"ial=",ial)
    
    
    
    ######
    ial=0
    leftover=list(range(Nindiv))
    #print("max,mean,std e11=",np.max(e11),np.mean(e11),np.std(e11))
    print("we start again from Ar=",Ar,"with e11 e00 max=",np.max(e11),np.max(e00))
    while Ar>1:#np.max(e11)>np.mean(e11)+np.std(e11) or ial<Nindiv-1: #np.max(e11)>np.mean(e11):
        j=np.argmax(e11) # taking the node with the max number of e11
        ial+=1
        ok=0
        if np.abs(Ar)<0.09:
            break
           
        if Immuned[j]==1 :
            Immuned[j]=0
            binage=int(Ageindiv[j]/10.)
            if binage>8:
                binage=8
            random.shuffle(leftover)
                
            for j2 in leftover:
                binage2=int(Ageindiv[j2]/10.)
                if binage2>8:
                    binage2=8
                if j2!=j and binage2==binage and Immuned[j2]==0 and e00[j2]>e10[j2]:
                    Immuned[j2]=1
                        
                    ###### compute r until it descreases
                    ncontact=np.zeros(Nindiv)
                    e00=np.zeros(Nindiv)
                    e11=np.zeros(Nindiv)
                    e10=np.zeros(Nindiv)
                    e01=np.zeros(Nindiv)
                    for i in range(0,Nindiv):
                        ihouse=Agent_house[i]
                        for  key in cluster_dict[ihouse]:
                            if i !=key:
                                ncontact[i]+=1
                            if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                e11[i]+=1
                            elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                e00[i]+=1
                            elif Immuned[i]==1 and Immuned[key]==0 :
                                e10[i]+=1

                        if Agent_school[i]>-1:
                            ischool=int(Agent_school[i])
                            for key in cluster_schools_dict[ischool]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1


                        if Agent_teen[i]>-1:
                            iteen=int(Agent_teen[i])
                            for key in cluster_teen_dict[iteen]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1

                        if Agent_eldery[i]>-1:
                            ieldery=int(Agent_eldery[i])
                            for key in cluster_eldery_dict[ieldery]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1
                                     #   print("el",i,key)

                        if Agent_work[i]>-1:
                            iwork=int(Agent_work[i])
                            for key in cluster_work_dict[iwork]:
                                if i !=key:
                                    ncontact[i]+=1
                                if Immuned[i]==1 and Immuned[key]==1 and i!=key :
                                    e11[i]+=1
                                elif Immuned[i]==0 and Immuned[key]==0 and i!=key  :
                                    e00[i]+=1
                                elif Immuned[i]==1 and Immuned[key]==0 :
                                    e10[i]+=1



                    e00g=np.sum(e00)/np.sum(ncontact)
                    e10g=(np.sum(e10)/np.sum(ncontact))
                    e01g=e10g
                    e11g=np.sum(e11)/np.sum(ncontact)
                    e=np.matrix( [[e00g,e01g],[e10g,e11g] ])
                    Arg=(np.trace(e)-np.sum(np.dot(e,e)))/(1.-np.sum(np.dot(e,e)))
                       # print(il,Arg)
                    if Arg<Ar:
                          #  print(j,"is replaced by",j2, Arg,Ar)
                        Ar=Arg
                        ok=1
                        leftover.remove(j)
                           # leftover.remove(j2)
                        break
                    else:
                        Immuned[j2]=0
            if ok!=1:
                Immuned[j]=1
     #           print(j,"we didn't found a better node",j2, Arg,Ar)
                break 
            
            
       # if ial%int(100)==0:
       #     print(" check ial=",ial, Ar,"fracind=",ial/Nindiv,"rnew/rini",Arg/Arr,"e11 max, mean,std=",np.max(e11),np.mean(e11),np.std(e11))
                
    print("we end up with Ar=",Ar,"with e11 e00 max=",np.max(e11),np.max(e00),"ial=",ial)
    
    print("matrix e=",e,"ial=",ial)
##################################
##################################
        
    for itest in Ntestperw:
         
        agent_tested=np.zeros(itest)
        ictested=0
        ### strategy: we test one person per house [i][0]
        for i in range(0,len(cluster_dict)):
            if ictested<itest:
                agent_tested[ictested]=cluster_dict[i][0]
                ictested+=1
        print("there are",nbhouse,"house and we make",itest,"tests",itest/Nindiv,"=frac indiv tested")    
        output2=open('r='+str("%.1f"%Ar)+'Transpernet_Reff_'+str("%.2f" % R)+'frac_vac'+str("%.2f" % ivac)+'Ntest='+str("%.2f" % itest)+'.csv', 'w')
        output2.write('network,bin_infected,bin_source,count'+'\n')


        output6=open('r='+str("%.1f"%Ar)+'Ninf_age_Reff_'+str("%.2f" % R)+'frac_vac'+str("%.2f" % ivac)+'Ntest='+str("%.2f" % itest)+'.csv', 'w')
        output6.write('isim,t,bin_age,ivac, Ninf'+'\n')
        


        
        Ninf=np.zeros((tmax,Nsim))
        Ninf_net=np.zeros((tmax,Nsim,3))
        Ninf_age=np.zeros((tmax,Nsim,9,2))
        #Nt0_inf=int(Nindiv*0.0011)

        Nnet=3 # 0 ran, 1 house, 2 occupation
        transevents=np.zeros((9,9,Nnet))
        Gene_event=np.zeros((28))

        #### house transmition as function of size
        trans_hsize=np.zeros((9,9,20))
        transtot_hsize=np.zeros((9,9,20))
        Ninf_detect=np.zeros((9,Nsim))
        Ninf_all=np.zeros((9,Nsim))

        for isim in range(0,Nsim):
            inte2=[i for i in range(Nindiv)]
            random.Random(isim).shuffle(inte2)
         
        
         #  
            ######## set the initial number of infected agents 
            Datet=np.zeros(Nindiv)
            Agent_health=np.zeros(Nindiv) #[0=s,1=i,2=r]
            Agent_quarantine=np.zeros(Nindiv) # 0 or 1 for quarantine 
            Agent_quarantine_t=np.zeros(Nindiv)     
            Datequa=np.zeros(Nindiv)
           # Ninf_detect=np.zeros(9)
           # Ninf_all=np.zeros(9)
             
            print("initial infected agent=",Nt0_inf,"over",Nindiv,"start sim",isim)
            #for i in range(0,Nt0_inf):
            ic0=0
            ip=-1
            while ic0<Nt0_inf:
                ip+=1
                ran=inte2[ip]
                if Immuned[ran]==0:
                    Agent_health[ran]=1
                    ic0+=1
                   
                
            Ninf[0,isim]=Nt0_inf
                
        
                ####### start the time loop 
            for iday in range(0,tmax):
                prop=np.zeros(Nindiv)
                
                    ##### random tests 
                if iday%7==0:        
                    for i in range(0,itest):
                        iagent=int(agent_tested[i]) # we test the same houses 
                        if (Agent_health[iagent]==1 and Datet[iagent]+tpositif_inf<=iday and random.uniform(0,1)<0.7 and
                                Datet[iagent]+tpositif_max>=iday):
                            Agent_quarantine_t[iagent]=1
                            Datequa[iagent]=iday
                            ihouse=Agent_house[iagent]
                            for key in cluster_dict[ihouse]:
                                if random.uniform(0,1)<0.8 and Agent_health[key]==1 and Datet[key]+tpositif_inf<=iday:
                                    Agent_quarantine_t[key]=1
                                    Datequa[key]=iday

                    
                                                        
################################################################################################################                  
################# start the loop over agents ####################################################################
################################################################################################################
                                                                    
                for i in range(0,Nindiv):
                    isymp=int(Agent_symp[i])
            
                    if Agent_health[i]==1 and Datet[i]+Agent_timerecovery[i]<=iday: 
                        Agent_health[i]=2         # the agent has recover 
         
            #### remove from quarantine due to random tests:
                    if Agent_quarantine_t[i]==1 and Datequa[i]+tquarantine<=iday:
                        Agent_quarantine_t[i]=0
                        Datequa[i]=0
                
                    binage_source=int(Agent_binage[i])
                    t_symp=Agent_timetosymptoms[i]  
            
                    # testing agent with symptomes
                   # if Agent_health[i]==1 and Datet[i]+1==iday and int(Agent_symp[i])==1 and random.uniform(0, 1)<0.5:
                   #     binage=int(Ageindiv[i]/10.)
                   #     if binage>8:
                   #         binage=8
                   #     Ninf_detect[binage,isim]+=1
                   # elif Agent_health[i]==1 and Datet[i]+1==iday:
                   #     binage=int(Ageindiv[i]/10.)
                   #     if binage>8:
                   #         binage=8
                   #     Ninf_all[binage,isim]+=1
                        
                         
                        
                        
        #### set this up for quarantine option for a given #% of agent that take a test
        #    isymp=int(Agent_symp[i])
        #    ran=random.uniform(0, 1)
        #    t_symp=Agent_timesymptoms[i]
        #    if Agent_health[i]==1 and Datet[i]+t_symp<iday and Agent_quarantine[i]==0 and ran<0.6 and isymp==1:
        #        if t_symp>test_insensitive_period:
        #            Agent_quarantine[i]=1                    
           
            
################################################################################################################        
################# start making the loop over contacts of the infected agent#####################################
################################################################################################################  
                   
                    
                    if (Agent_health[i]==1 and prop[i]==0 and Agent_quarantine_t[i]==0 and Immuned[i]==0) or \
                        (Agent_health[i]==1 and prop[i]==0 and Agent_quarantine_t[i]==1 and random.uniform(0, 1)<0.3) or \
                         (Agent_health[i]==1 and prop[i]==0 and Agent_quarantine_t[i]==0 and Immuned[i]==1 and random.uniform(0, 1)<ptvac ) :
                         
                       # print(iday,"after",i,Agent_health[i], prop[i], Agent_quarantine_t[i], Immuned[i])        
                            
                          ######## make a loop over random pp  
                        Nran=int(Agent_rancont[i])
                        for iran in range(0,Nran):
                            icont=int(random.uniform(0,1)*Nindiv)
                    
                            binage=int(Ageindiv[icont]/10.)
                            ran=random.uniform(0, 1)
                            if binage>8:
                                  binage=8
                            lam=R*Bran*A[isymp]*Susceptibility[binage]*gamma.pdf(iday-0.5,a,Datet[i],scale)/avdailyinteract
                            Ptrans=1.-np.exp(-lam)
                            
                            if (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==0) or \
                                (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==1 and random.uniform(0, 1)>pivac):
                                Agent_health[icont]=1
                                Datet[icont]=iday
                                prop[icont]=1
                                Ninf[iday,isim]+=1
                                Ninf_net[iday,isim,0]+=1
                                #Ninf_age[iday,isim,binage]+=1
                                ivac=int(Immuned[icont])
                                Ninf_age[iday,isim,binage,ivac]+=1
                                transevents[binage,binage_source,0]+=1
                      #          Gene_event[int(iday-Datet[i])]+=1
                             
                                
                                
            ######## loop over the house the agent belong to
                        ihouse=Agent_house[i]
                        for key in cluster_dict[ihouse]:
                            icont=key
                            ran=random.uniform(0, 1)
                            binage=int(Ageindiv[icont]/10.)
                            if binage>8:
                                binage=8
                                     
                            lam=R*Bhome*A[isymp]*Susceptibility[binage]*gamma.pdf(iday-0.5,a,Datet[i],scale)/avdailyinteract
                            Ptrans=1.-np.exp(-lam*3./len(cluster_dict[ihouse])**1.2)  
                    
                       #     if (Agent_health[icont]==0):
                       #         transtot_hsize[binage,binage_source,len(cluster_dict[ihouse])]+=1
                    
                            if (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==0) or \
                                (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==1 and random.uniform(0, 1)>pivac):
                                Agent_health[icont]=1
                                Datet[icont]=iday
                                prop[icont]=1
                                Ninf[iday,isim]+=1
                                Ninf_net[iday,isim,1]+=1
                               # Ninf_age[iday,isim,binage]+=1
                                ivac=int(Immuned[key])
                                Ninf_age[iday,isim,binage,ivac]+=1
                                transevents[binage,binage_source,1]+=1
                         #       Gene_event[int(iday-Datet[i])]+=1
                         #       trans_hsize[binage,binage_source,len(cluster_dict[ihouse])]+=1
                                     #countReff[i]+=1
                        
            ######## loop over school children:  
                        if Agent_school[i]>-1:
                             ischool=int(Agent_school[i])
                             for key in cluster_schools_dict[ischool]:
                                halftime=(random.uniform(0, 1))
                                if halftime>Nchildren/Nadult*1.8: 
                                    icont=key  
                                    binage=int(Ageindiv[icont]/10.)
                                    if binage>8:
                                        binage=8
                                    ran=random.uniform(0, 1)
                                    lam=R*Bother*A[isymp]*Susceptibility[binage]*gamma.pdf(iday-0.5,a,Datet[i],scale)/avdailyinteract
                                    Ptrans=1.-np.exp(-lam)
                                    if (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==0) or \
                                       (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==1 and random.uniform(0, 1)>pivac):
                                        Agent_health[icont]=1
                                        Datet[icont]=iday
                                        prop[icont]=1
                                        Ninf[iday,isim]+=1
                                        Ninf_net[iday,isim,2]+=1
                                        ivac=int(Immuned[key])
                                        Ninf_age[iday,isim,binage,ivac]+=1
                                        #Ninf_age[iday,isim,binage]+=1
                                        transevents[binage,binage_source,2]+=1
                                        
                                #        Gene_event[int(iday-Datet[i])]+=1
                                #countReff[i]+=1
                                
            ######## loop over school teen:  
                        if Agent_teen[i]>-1:
                            iteen=int(Agent_teen[i])
                            for key in cluster_teen_dict[iteen]:
                                halftime=(random.uniform(0, 1))
                                if halftime>Nteen/Nadult*1.7:
                                    icont=key  
                                    binage=int(Ageindiv[icont]/10.)
                                    if binage>8:
                                        binage=8
                                    ran=random.uniform(0, 1)
                                    lam=R*Bother*A[isymp]*Susceptibility[binage]*gamma.pdf(iday-0.5,a,Datet[i],scale)/avdailyinteract
                                    Ptrans=1.-np.exp(-lam)
                                    if (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==0) or \
                                       (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==1 and random.uniform(0, 1)>pivac):
                                        Agent_health[icont]=1
                                        Datet[icont]=iday
                                        prop[icont]=1
                                        Ninf[iday,isim]+=1  
                                        Ninf_net[iday,isim,2]+=1
                                        #Ninf_age[iday,isim,binage]+=1
                                        ivac=int(Immuned[key])
                                        Ninf_age[iday,isim,binage,ivac]+=1
                                        transevents[binage,binage_source,2]+=1
                                 #       Gene_event[int(iday-Datet[i])]+=1
                                #countReff[i]+=1
                                
            ######## loop over eldery net:  
                        if Agent_eldery[i]>-1:
                            ieldery=int(Agent_eldery[i])
                            for key in cluster_eldery_dict[ieldery]:
                                halftime=(random.uniform(0, 1))
                                if halftime>0.72:
                                    icont=key  
                                    binage=int(Ageindiv[icont]/10.)
                                    if binage>8:
                                        binage=8
                                    ran=random.uniform(0, 1)
                                    lam=R*Bother*A[isymp]*Susceptibility[binage]*gamma.pdf(iday-0.5,a,Datet[i],scale)/avdailyinteract
                                    Ptrans=1.-np.exp(-lam)
                                    if (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==0) or \
                                        (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==1 and random.uniform(0, 1)>pivac):
                                        Agent_health[icont]=1
                                        Datet[icont]=iday
                                        prop[icont]=1
                                        Ninf[iday,isim]+=1        
                                        Ninf_net[iday,isim,2]+=1
                                        #Ninf_age[iday,isim,binage]+=1
                                        ivac=int(Immuned[key])
                                        Ninf_age[iday,isim,binage,ivac]+=1
                                        transevents[binage,binage_source,2]+=1
                                #        Gene_event[int(iday-Datet[i])]+=1
                                #countReff[i]+=1
                                
            ######## loop over work:  
                        if Agent_work[i]>-1:
                             iwork=int(Agent_work[i])
                             for key in cluster_work_dict[iwork]:
                                halftime=(random.uniform(0, 1))#round(random.uniform(0, 1))
                                if halftime>0.82:
                                    icont=key
                                    binage=int(Ageindiv[icont]/10.)
                                    if binage>8:
                                        binage=8
                                    ran=random.uniform(0, 1)
                                    lam=R*Bother*A[isymp]*Susceptibility[binage]*gamma.pdf(iday-0.5,a,Datet[i],scale)/avdailyinteract
                                    Ptrans=1.-np.exp(-lam)
                                    if (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==0) or \
                                       (ran<Ptrans and Agent_health[icont]==0 and Immuned[icont]==1 and random.uniform(0, 1)>pivac):
                                        Agent_health[icont]=1
                                        Datet[icont]=iday
                                        prop[icont]=1
                                        Ninf[iday,isim]+=1
                                        Ninf_net[iday,isim,2]+=1
                                        ivac=int(Immuned[key])
                                        Ninf_age[iday,isim,binage,ivac]+=1
                                        transevents[binage,binage_source,2]+=1
                                  #      Gene_event[int(iday-Datet[i])]+=1
                                #countReff[i]+=1

#                output.write(str(isim)+ ', ' + str(iday)+', '+str(Ninf[iday,isim])+'\n')
        
                
            s=np.cumsum(Ninf[:,isim])        
            print("sim",isim,"has Ninfected=",s[tmax-1],"at T=",tmax-1,"frac_inf=",s[tmax-1]/Nindiv,"Reff=",R,"Nbtest=",itest)        
    
    

       
#        for i in range(0,Nsim): 
#            for j in range(0,len(t)):
#                for inet in range(0,3):
#                    output5.write(str(i)+ ', ' + str(j)+', '+ str(inet)+', '+str(Ninf_net[j,i,inet])+'\n')
            
        for i in range(0,Nsim): 
            for j in range(0,len(t)):
#                output.write(str(i)+ ', ' + str(j)+', '+str(Ninf[j,i])+'\n')
                for iage in range(0,9):
                    for ivac in range(0,2):
                        output6.write(str(i)+ ', ' + str(j)+', '+ str(iage)+', '+str(ivac)+', '+str(Ninf_age[j,i,iage,ivac])+'\n')

        for inet in range(0,3):
            for j1 in range(0,9):
                for j2 in range(0,9):
                    output2.write(str(inet)+ ', ' + str(j1)+', '+ str(j2)+', '+str(transevents[j1,j2,inet])+'\n')

#        for itime in range(0,len(Gene_event)):
#            output3.write(str(itime)+ ', ' +str(Gene_event[itime])+'\n')
  
#        for isize in range(0,20):
#            for j1 in range(0,9):
#                for j2 in range(0,9):
#                    output4.write(str(isize)+ ', ' + str(j1)+', '+ str(j2)+', '+str(trans_hsize[j1,j2,isize])+', '+str(transtot_hsize[j1,j2,isize])+'\n')

#        for isim in range(0,Nsim):
#            for i in range(0,9):
#                output7.write(str(isim)+", " +str(i)+ ", " + str(Ninf_all[i,isim])+", " + str(Ninf_detect[i,isim]) + ", " + str(Nperage[i])+"\n")
        
                    
#        output.close()   
        output2.close()
#        output3.close()
#        output4.close()
#        output5.close()
        output6.close()
#        output7.close()
print(time.time()- start_time, "seconds")








