# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 00:37:45 2020

@author: alf11
"""

import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns


# Import CSV
FDX_data = pd.read_csv("FDX.csv") 
SPY_data = pd.read_csv("SPY.csv") 

# Create Dataframes
FDX_df = df = pd.DataFrame(FDX_data)
SPY_df = df = pd.DataFrame(SPY_data)

# Put the return column in a list

FDX_return = FDX_df["Return"]
SPY_return = SPY_df["Return"]

# Make a new list for +,-

FDX_true = []
SPY_true = []

# Look at the return column and assign true value +,-

for e in FDX_return:
    if e >= 0:
        FDX_true.append("+")
    else :
        FDX_true.append("-")
        
for e in SPY_return:
    if e >= 0:
        SPY_true.append("+")
    else :
        SPY_true.append("-")


# Add the new columns back to the dataframe

FDX_df["True Label"] = FDX_true
SPY_df["True Label"] = SPY_true

# Make training dataframes years 1 - 3

FDX_123 = FDX_df[(FDX_df["Year"] < 2018)]
SPY_123 = SPY_df[(SPY_df["Year"] < 2018)]

# Total days, total true labels

day_count = FDX_123.shape[0]
FDX_123_True = FDX_123["True Label"].value_counts()
SPY_123_True = SPY_123["True Label"].value_counts()

print("Question 1.1:")
print("Days in years 1-3", day_count)
print("FDX :")
print(FDX_123_True)
print("SPY :")
print(SPY_123_True)

print("Question 1.2")
print("Probability of up day FDX :", round((394/day_count),4))
print("Probability of up day SPY :", round((404/day_count),4))


# Create lists for values after k negatives, positives
        

k1_neg_fdx = []
k2_neg_fdx = []
k3_neg_fdx = []
k1_neg_spy = [] 
k2_neg_spy = []
k3_neg_spy = []

k1_pos_fdx = []
k2_pos_fdx = []
k3_pos_fdx = []
k1_pos_spy = []
k2_pos_spy = []
k3_pos_spy = []   




#Create list of labels years 1 - 3
True_FDX_train = FDX_true[0:755]
True_SPY_train = SPY_true[0:755]

#For k = 1, enter the next value into pos or neg list

i = 0
for e in True_FDX_train:
    if i < 753:
        i = i+1
        if e == "-":
            k1_neg_fdx.append(True_FDX_train[i])
        else:
            k1_pos_fdx.append(True_FDX_train[i])
        
        
a = 0
for e in True_SPY_train:
    if a < 753:
        a = a+1
        if e == "-":
            k1_neg_spy.append(True_FDX_train[a])
        else:
            k1_pos_spy.append(True_SPY_train[a])
            
        


#Divide the neg count in k1 list by day count -1

print("Question 1.3")
print("Probability of up day FDX k1:", round((k1_neg_fdx.count("+"))/len(k1_neg_fdx),4))
print("Probability of up day SPY k1:", round((k1_neg_spy.count("+"))/len(k1_neg_spy),4))

#For k = 2, count the number of doubles. append the 3rd value to pos/neg lists

fdx_k2 = 0
ii = 0
for e in True_FDX_train:
    if ii < 752:
        if True_FDX_train[ii] == True_FDX_train[ii+1]:
            fdx_k2 = fdx_k2 + 1
            if True_FDX_train[ii] == "-":
                k2_neg_fdx.append(True_FDX_train[ii+2])
            else:
                k2_pos_fdx.append(True_FDX_train[ii+2])
        ii = ii+1
        
        
spy_k2 = 0
aa = 0
for e in True_SPY_train:
    if aa < 752:
        if True_SPY_train[aa] == True_SPY_train[aa+1]:
            spy_k2 = spy_k2 + 1
            if True_SPY_train[aa] == "-":
                k2_neg_spy.append(True_SPY_train[aa+2])
            else:
                k2_pos_spy.append(True_SPY_train[aa+2])
        aa = aa +1
                
print("Probability of up day FDX k2:", round((k2_neg_fdx.count("+"))/len(k2_neg_fdx),4))
print("Probability of up day SPY k2:", round((k2_neg_spy.count("+"))/len(k2_neg_spy),4))

   
#For k = 3, count the number of triples. append the 4th value to pos/neg lists

fdx_k3 = 0
iii = 0
for e in True_FDX_train:
    if iii < 752:
        if True_FDX_train[iii] == True_FDX_train[iii+1] == True_FDX_train[iii+2]:
            fdx_k3 = fdx_k3 + 1
            if True_FDX_train[iii] == "-":
                k3_neg_fdx.append(True_FDX_train[iii+3])
            else:
                k3_pos_fdx.append(True_FDX_train[iii+3])
        iii = iii+1
        
        
spy_k3 = 0
aaa = 0
for e in True_SPY_train:
    if aaa < 752:
        if True_SPY_train[aaa] == True_SPY_train[aaa+1] == True_SPY_train[aaa+2]:
            spy_k3 = spy_k3 + 1
            if True_SPY_train[aaa] == "-":
                k3_neg_spy.append(True_SPY_train[aaa+3])
            else:
                k3_pos_spy.append(True_SPY_train[aaa+3])
        aaa = aaa +1

print("Probability of up day FDX k3:", round((k3_neg_fdx.count("+"))/len(k3_neg_fdx),4))
print("Probability of up day SPY k3:", round((k3_neg_spy.count("+"))/len(k3_neg_spy),4))
  

print("Question 1.4")
print("Probability of down day FDX k1:", round((k1_pos_fdx.count("-"))/len(k1_pos_fdx),4))
print("Probability of down day SPY k1:", round((k1_pos_spy.count("-"))/len(k1_pos_spy),4))
                
print("Probability of down day FDX k2:", round((k2_pos_fdx.count("-"))/len(k2_pos_fdx),4))
print("Probability of down day SPY k2:", round((k2_pos_spy.count("-"))/len(k2_pos_spy),4))


print("Probability of down day FDX k3:", round((k3_pos_fdx.count("-"))/len(k3_pos_fdx),4))
print("Probability of down day SPY k3:", round((k3_pos_spy.count("-"))/len(k3_pos_spy),4))
  

# Make test dataframes years 4-5

FDX_45 = FDX_df[(FDX_df["Year"] > 2017)]
SPY_45 = SPY_df[(SPY_df["Year"] > 2017)]

# Calculate the probabilites for all other possibilities in k2 FDX

np_fdx = []
pn_fdx = []

b = 0
for e in True_FDX_train:
    if b < 752:
        if True_FDX_train[b] != True_FDX_train[b+1]:
            if True_FDX_train[b] == "-":
                np_fdx.append(True_FDX_train[b+2])
            else:
                pn_fdx.append(True_FDX_train[b+2])
        b = b+1

print("Question 2 probabilities:" )
print("Probability of up day FDX np:", round((np_fdx.count("+"))/len(np_fdx),2))
print("Probability of up day FDX pn:", round((pn_fdx.count("+"))/len(pn_fdx),2))

#Store Values based on probablilites

fdx_pp = "+"
fdx_pn = "+"
fdx_np = "-"
fdx_nn = "+"


#Create list of labels for FDX W2

fdx_w2 = []

#Checks prior true labels and assigns next label 
kk2 = 755
while kk2 < 1258:
    if FDX_true[kk2-2] == FDX_true[kk2-1]:
        if FDX_true[kk2-2] == "+":
            fdx_w2.append(fdx_pp)
        else:
            fdx_w2.append(fdx_nn)
    elif FDX_true[kk2-2] == "+":
        fdx_w2.append(fdx_pn)
    else:
        fdx_w2.append(fdx_np)
    kk2 = kk2+1


# Calculate the probabilites for all other possibilities in k2 SPY

np_spy = []
pn_spy = []

bb = 0
for e in True_SPY_train:
    if bb < 752:
        if True_SPY_train[bb] != True_SPY_train[bb+1]:
            if True_SPY_train[bb] == "-":
                np_spy.append(True_SPY_train[bb+2])
            else:
                pn_spy.append(True_SPY_train[bb+2])
        bb = bb+1

print("Probability of up day SPY np:", round((np_spy.count("+"))/len(np_spy),2))
print("Probability of up day SPY pn:", round((pn_spy.count("+"))/len(pn_spy),2))


spy_pp = "-"
spy_pn = "+"
spy_np = "+"
spy_nn = "+"


#Create list of labels for SPY W2

spy_w2 = []

xx2 = 755
while xx2 < 1258:
    if SPY_true[xx2-2] == SPY_true[xx2-1]:
        if SPY_true[xx2-2] == "+":
            spy_w2.append(spy_pp)
        else:
            spy_w2.append(spy_nn)
    elif SPY_true[xx2-2] == "+":
        spy_w2.append(spy_pn)
    else:
        spy_w2.append(spy_np)
    xx2 = xx2+1

# Compute accuracies, compare true label in year 4&5 to W2 computed label

fdx_test_true = FDX_true[755:]

w2_tp_fdx = 0
w2_fp_fdx = 0
w2_tn_fdx = 0
w2_fn_fdx = 0

u = 0
for e in fdx_test_true:
    if fdx_test_true[u] == fdx_w2[u]:
        if fdx_test_true[u] =="+":
            w2_tp_fdx = w2_tp_fdx +1
        else:
            w2_tn_fdx = w2_tn_fdx+1
    else:
        if fdx_w2[u] == "+":
            w2_fp_fdx = w2_fp_fdx +1
        else:
            w2_fn_fdx = w2_fn_fdx+1
    u = u+1
    
print("FDX w2 TP:", w2_tp_fdx, "FP", w2_fp_fdx,"TN", w2_tn_fdx , "FN", w2_fn_fdx)



spy_test_true = SPY_true[755:]

w2_tp_spy = 0
w2_fp_spy = 0
w2_tn_spy = 0
w2_fn_spy = 0

w = 0
for e in spy_test_true:
    if spy_test_true[w] == spy_w2[w]:
        if spy_test_true[w] =="+":
            w2_tp_spy = w2_tp_spy +1
        else:
            w2_tn_spy = w2_tn_spy+1
    else:
        if spy_w2[w] == "+":
            w2_fp_spy = w2_fp_spy +1
        else:
            w2_fn_spy = w2_fn_spy+1
    w = w+1

print("SPY w2 TP:", w2_tp_spy, "FP", w2_fp_spy,"TN", w2_tn_spy , "FN", w2_fn_spy)                


# Calculate the probabilites for all other possibilities in k3 FDX

fdx_ppn = []
fdx_pnp = []
fdx_pnn = []
fdx_nnp = []
fdx_npn = []
fdx_npp = []



bb = 0
for e in True_FDX_train:
    if bb < 752:
        if True_FDX_train[bb] == True_FDX_train[bb+1] != True_FDX_train[bb+2]:
            if True_FDX_train[bb] == "-":
                fdx_nnp.append(True_FDX_train[bb+3])
            else:
                fdx_ppn.append(True_FDX_train[bb+3])
        elif True_FDX_train[bb] != True_FDX_train[bb+1] == True_FDX_train[bb+2]:
            if True_FDX_train[bb] == "-":
                fdx_npp.append(True_FDX_train[bb+3])
            else:
                fdx_pnn.append(True_FDX_train[bb+3]) 
        elif True_FDX_train[bb] == True_FDX_train[bb+2] != True_FDX_train[bb+1]:
            if True_FDX_train[bb] == "-":
                fdx_npn.append(True_FDX_train[bb+3])
            else:
                fdx_pnp.append(True_FDX_train[bb+3]) 
        bb = bb+1



print("Probability of up day FDX pnn:", round((fdx_pnn.count("+"))/len(fdx_pnn),2))
print("Probability of up day FDX ppn:", round((fdx_ppn.count("+"))/len(fdx_ppn),2))
print("Probability of up day FDX pnp:", round((fdx_pnp.count("+"))/len(fdx_pnp),2))
print("Probability of up day FDX nnp:", round((fdx_nnp.count("+"))/len(fdx_nnp),2))
print("Probability of up day FDX npn:", round((fdx_npn.count("+"))/len(fdx_npn),2))
print("Probability of up day FDX npp:", round((fdx_npp.count("+"))/len(fdx_npp),2))

ppp_fdx = "-"
ppn_fdx = "+"
pnp_fdx = "+"
pnn_fdx = "+"
nnn_fdx = "+"
nnp_fdx = "-"
npn_fdx = "-"
npp_fdx = "+"


#Create list of labels for FDX W3

fdx_w3 = []

#Checks prior true labels and assigns next label 
kk3 = 755
while kk3 < 1258:
    if FDX_true[kk3-3] == FDX_true[kk3-2] == FDX_true[kk3-1] :
        if FDX_true[kk3-2] == "+":
            fdx_w3.append(ppp_fdx)
        else:
            fdx_w3.append(nnn_fdx)
    elif FDX_true[kk3-3] == FDX_true[kk3-2] != FDX_true[kk3-1]:
        if FDX_true[kk3-3] == "+":
            fdx_w3.append(ppn_fdx)
        else:
            fdx_w3.append(nnp_fdx)
    elif FDX_true[kk3-3] != FDX_true[kk3-2] == FDX_true[kk3-1]:
        if FDX_true[kk3-3] == "+":
            fdx_w3.append(pnn_fdx)
        else:
            fdx_w3.append(npp_fdx)
    elif FDX_true[kk3-3] == FDX_true[kk3-1] !=FDX_true[kk3-2]:
        if FDX_true[kk3-3] == "+":
            fdx_w3.append(pnp_fdx)
        else:
            fdx_w3.append(npn_fdx)       
    kk3 = kk3+1

# Compute accuracies, compare true label in year 4&5 to W3 computed label


w3_tp_fdx = 0
w3_fp_fdx = 0
w3_tn_fdx = 0
w3_fn_fdx = 0

u = 0
for e in fdx_test_true:
    if fdx_test_true[u] == fdx_w3[u]:
        if fdx_test_true[u] =="+":
            w3_tp_fdx = w3_tp_fdx +1
        else:
            w3_tn_fdx = w3_tn_fdx+1
    else:
        if fdx_w3[u] == "+":
            w3_fp_fdx = w3_fp_fdx +1
        else:
            w3_fn_fdx = w3_fn_fdx+1
    u = u+1
    
print("FDX w3 TP:", w3_tp_fdx, "FP", w3_fp_fdx,"TN", w3_tn_fdx , "FN", w3_fn_fdx)

# Calculate the probabilites for all other possibilities in k3 FDX

spy_ppn = []
spy_pnp = []
spy_pnn = []
spy_nnp = []
spy_npn = []
spy_npp = []



uu = 0
for e in True_SPY_train:
    if uu < 752:
        if True_SPY_train[uu] == True_SPY_train[uu+1] != True_SPY_train[uu+2]:
            if True_SPY_train[uu] == "-":
                spy_nnp.append(True_SPY_train[uu+3])
            else:
                spy_ppn.append(True_SPY_train[uu+3])
        elif True_SPY_train[uu] != True_SPY_train[uu+1] == True_SPY_train[uu+2]:
            if True_SPY_train[uu] == "-":
                spy_npp.append(True_SPY_train[uu+3])
            else:
                spy_pnn.append(True_SPY_train[uu+3]) 
        elif True_SPY_train[uu] == True_SPY_train[uu+2] != True_SPY_train[uu+1]:
            if True_SPY_train[uu] == "-":
                spy_npn.append(True_SPY_train[uu+3])
            else:
                spy_pnp.append(True_SPY_train[uu+3]) 
        uu = uu+1





print("Probability of up day SPY pnn:", round((spy_pnn.count("+"))/len(spy_pnn),2))
print("Probability of up day SPY ppn:", round((spy_ppn.count("+"))/len(spy_ppn),2))
print("Probability of up day SPY pnp:", round((spy_pnp.count("+"))/len(spy_pnp),2))
print("Probability of up day SPY nnp:", round((spy_nnp.count("+"))/len(spy_nnp),2))
print("Probability of up day SPY npn:", round((spy_npn.count("+"))/len(spy_npn),2))
print("Probability of up day SPY npp:", round((spy_npp.count("+"))/len(spy_npp),2))

ppp_spy = "-"
ppn_spy = "+"
pnp_spy = "-"
pnn_spy = "+"
nnn_spy = "+"
nnp_spy = "+"
npn_spy = "+"
npp_spy = "-"


#Create list of labels for SPY W2

spy_w3 = []

#Checks prior true labels and assigns next label 
kk3 = 755
while kk3 < 1258:
    if SPY_true[kk3-3] == SPY_true[kk3-2] == SPY_true[kk3-1] :
        if SPY_true[kk3-2] == "+":
            spy_w3.append(ppp_spy)
        else:
            spy_w3.append(nnn_spy)
    elif SPY_true[kk3-3] == SPY_true[kk3-2] != SPY_true[kk3-1]:
        if SPY_true[kk3-3] == "+":
            spy_w3.append(ppn_spy)
        else:
            spy_w3.append(nnp_spy)
    elif SPY_true[kk3-3] != SPY_true[kk3-2] == SPY_true[kk3-1]:
        if SPY_true[kk3-3] == "+":
            spy_w3.append(pnn_spy)
        else:
            spy_w3.append(npp_spy)
    elif SPY_true[kk3-3] == SPY_true[kk3-1] !=SPY_true[kk3-2]:
        if SPY_true[kk3-3] == "+":
            spy_w3.append(pnp_spy)
        else:
            spy_w3.append(npn_spy)       
    kk3 = kk3+1

# Compute accuracies, compare true label in year 4&5 to W3 computed label


w3_tp_spy = 0
w3_fp_spy = 0
w3_tn_spy = 0
w3_fn_spy = 0

u = 0
for e in spy_test_true:
    if spy_test_true[u] == spy_w3[u]:
        if spy_test_true[u] =="+":
            w3_tp_spy = w3_tp_spy +1
        else:
            w3_tn_spy = w3_tn_spy+1
    else:
        if spy_w3[u] == "+":
            w3_fp_spy = w3_fp_spy +1
        else:
            w3_fn_spy = w3_fn_spy+1
    u = u+1
    
print("SPY w3 TP:", w3_tp_spy, "FP", w3_fp_spy,"TN", w3_tn_spy , "FN", w3_fn_spy)

# Calculate the probabilites for all other possibilities in k4 FDX


fdx_pppp = []
fdx_pppn = []
fdx_ppnn = []
fdx_ppnp = []
fdx_pnpp = []
fdx_pnpn = []
fdx_pnnp = []
fdx_pnnn = []
fdx_nnnn = []
fdx_nnnp = []
fdx_nnpp = []
fdx_npnp = []
fdx_nppn = []
fdx_nnpn = []
fdx_nppp = []
fdx_npnn = []


bbb = 0
for e in True_FDX_train:
    if bbb < 751:
        if True_FDX_train[bbb] == True_FDX_train[bbb+1] == True_FDX_train[bbb+2]== True_FDX_train[bbb+3]:
            if True_FDX_train[bbb] == "-":
                fdx_nnnn.append(True_FDX_train[bbb+4])
            else:
                fdx_pppp.append(True_FDX_train[bbb+4])
        elif True_FDX_train[bbb] == True_FDX_train[bbb+1] == True_FDX_train[bbb+2] != True_FDX_train[bbb+3]:
            if True_FDX_train[bbb] == "-":
                fdx_nnnp.append(True_FDX_train[bbb+4])
            else:
                fdx_pppn.append(True_FDX_train[bbb+4])
        elif True_FDX_train[bbb] == True_FDX_train[bbb+1] != True_FDX_train[bbb+2] == True_FDX_train[bbb+3]:
            if True_FDX_train[bbb] == "-":
                fdx_nnpp.append(True_FDX_train[bbb+4])
            else:
                fdx_ppnn.append(True_FDX_train[bbb+4]) 
        elif True_FDX_train[bbb] == True_FDX_train[bbb+1] == True_FDX_train[bbb+3] != True_FDX_train[bbb+2]:
            if True_FDX_train[bbb] == "-":
                fdx_nnpn.append(True_FDX_train[bbb+4])
            else:
                fdx_ppnp.append(True_FDX_train[bbb+4]) 
        elif True_FDX_train[bbb] == True_FDX_train[bbb+2] == True_FDX_train[bbb+3] != True_FDX_train[bbb+1]:
            if True_FDX_train[bbb] == "-":
                fdx_npnn.append(True_FDX_train[bbb+4])
            else:
                fdx_pnpp.append(True_FDX_train[bbb+4]) 
        elif True_FDX_train[bbb] == True_FDX_train[bbb+2] != True_FDX_train[bbb+3] == True_FDX_train[bbb+1]:
            if True_FDX_train[bbb] == "-":
                fdx_npnp.append(True_FDX_train[bbb+4])
            else:
                fdx_pnpn.append(True_FDX_train[bbb+4])
        elif True_FDX_train[bbb+1] == True_FDX_train[bbb+2] == True_FDX_train[bbb+3] != True_FDX_train[bbb]:
            if True_FDX_train[bbb] == "-":
                fdx_nppp.append(True_FDX_train[bbb+4])
            else:
                fdx_pnnn.append(True_FDX_train[bbb+4])
        elif True_FDX_train[bbb] == True_FDX_train[bbb+3] != True_FDX_train[bbb+2] == True_FDX_train[bbb+1]:
            if True_FDX_train[bbb] == "-":
                fdx_nppn.append(True_FDX_train[bbb+4])
            else:
                fdx_pnnp.append(True_FDX_train[bbb+4])
        bbb = bbb+1

print("Probability of up day FDX pppp:", round((fdx_pppp.count("+"))/len(fdx_pppp),2))
print("Probability of up day FDX pppn:", round((fdx_pppn.count("+"))/len(fdx_pppn),2))
print("Probability of up day FDX ppnn:", round((fdx_ppnn.count("+"))/len(fdx_ppnn),2))
print("Probability of up day FDX ppnp:", round((fdx_ppnp.count("+"))/len(fdx_ppnp),2))
print("Probability of up day FDX pnpp:", round((fdx_pnpp.count("+"))/len(fdx_pnpp),2))
print("Probability of up day FDX pnpn:", round((fdx_pnpn.count("+"))/len(fdx_pnpn),2))
print("Probability of up day FDX pnnp:", round((fdx_pnnp.count("+"))/len(fdx_pnnp),2))
print("Probability of up day FDX pnnn:", round((fdx_pnnn.count("+"))/len(fdx_pnnn),2))

print("Probability of up day FDX nnnn:", round((fdx_nnnn.count("+"))/len(fdx_nnnn),2))
print("Probability of up day FDX nnnp:", round((fdx_nnnp.count("+"))/len(fdx_nnnp),2))
print("Probability of up day FDX nnpp:", round((fdx_nnpp.count("+"))/len(fdx_nnpp),2))
print("Probability of up day FDX nnpn:", round((fdx_nnpn.count("+"))/len(fdx_nnpn),2))
print("Probability of up day FDX nppp:", round((fdx_nppp.count("+"))/len(fdx_nppp),2))
print("Probability of up day FDX nppn:", round((fdx_nppn.count("+"))/len(fdx_nppn),2))
print("Probability of up day FDX npnp:", round((fdx_npnp.count("+"))/len(fdx_npnp),2))
print("Probability of up day FDX npnn:", round((fdx_npnn.count("+"))/len(fdx_npnn),2))

pppp_fdx = "-"
pppn_fdx = "-"
ppnn_fdx = "+"
ppnp_fdx = "-"
pnpp_fdx = "+"
pnpn_fdx = "+"
pnnp_fdx = "-"
pnnn_fdx = "+"
nnnn_fdx = "+"
nnnp_fdx = "-"
nnpp_fdx = "+"
nnpn_fdx = "-"
npnn_fdx = "-"
nppp_fdx = "-"
npnp_fdx = "+"
nppn_fdx = "+"


#Create list of labels for FDX W3

fdx_w4 = []

#Checks prior true labels and assigns next label 
kk4 = 755
while kk4 < 1258:
    if FDX_true[kk4-4] == FDX_true[kk4-3] == FDX_true[kk4-2] == FDX_true[kk4-1]:
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(pppp_fdx)
        else:
            fdx_w4.append(nnnn_fdx)
    elif FDX_true[kk4-4] == FDX_true[kk4-3] == FDX_true[kk4-2] != FDX_true[kk4-1]:
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(pppn_fdx)
        else:
            fdx_w4.append(nnnp_fdx)
    elif FDX_true[kk4-4] == FDX_true[kk4-3] != FDX_true[kk4-2] == FDX_true[kk4-1]:
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(ppnn_fdx)
        else:
            fdx_w4.append(nnpp_fdx)
    elif FDX_true[kk4-4] == FDX_true[kk4-3] == FDX_true[kk4-1] !=FDX_true[kk4-2]:
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(ppnp_fdx)
        else:
            fdx_w4.append(nnpn_fdx)      
    elif FDX_true[kk4-3] == FDX_true[kk4-1] == FDX_true[kk4-2] != FDX_true[kk4-4] :
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(pnnn_fdx)
        else:
            fdx_w4.append(nppp_fdx)  
    elif FDX_true[kk4-4] == FDX_true[kk4-2] !=FDX_true[kk4-1] == FDX_true[kk4-3] :
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(pnpn_fdx)
        else:
            fdx_w4.append(npnp_fdx)  
    elif FDX_true[kk4-4] == FDX_true[kk4-2] == FDX_true[kk4-1] != FDX_true[kk4-3] :
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(pnpp_fdx)
        else:
            fdx_w4.append(npnn_fdx) 
    elif FDX_true[kk4-4] == FDX_true[kk4-1] != FDX_true[kk4-2] == FDX_true[kk4-3] :
        if FDX_true[kk4-4] == "+":
            fdx_w4.append(pnnp_fdx)
        else:
            fdx_w4.append(nppn_fdx)   
    kk4 = kk4+1
    
# Compute accuracies, compare true label in year 4&5 to W4 computed label

print(len(fdx_test_true))
print(len(fdx_w4))


w4_tp_fdx = 0
w4_fp_fdx = 0
w4_tn_fdx = 0
w4_fn_fdx = 0

uu = 0
for e in fdx_test_true:
    if fdx_test_true[uu] == fdx_w4[uu]:
        if fdx_test_true[uu] =="+":
            w4_tp_fdx = w4_tp_fdx +1
        else:
            w4_tn_fdx = w4_tn_fdx+1
    else:
        if fdx_w4[uu] == "+":
            w4_fp_fdx = w4_fp_fdx +1
        else:
            w4_fn_fdx = w4_fn_fdx+1
    uu = uu+1
    
print("FDX w4 TP:", w4_tp_fdx, "FP", w4_fp_fdx,"TN", w4_tn_fdx , "FN", w4_fn_fdx)


# Calculate the probabilites for all other possibilities in k4 SPY


spy_pppp = []
spy_pppn = []
spy_ppnn = []
spy_ppnp = []
spy_pnpp = []
spy_pnpn = []
spy_pnnp = []
spy_pnnn = []
spy_nnnn = []
spy_nnnp = []
spy_nnpp = []
spy_npnp = []
spy_nppn = []
spy_nnpn = []
spy_nppp = []
spy_npnn = []


bbbb = 0
for e in True_SPY_train:
    if bbbb < 751:
        if True_SPY_train[bbbb] == True_SPY_train[bbbb+1] == True_SPY_train[bbbb+2]== True_SPY_train[bbbb+3]:
            if True_SPY_train[bbbb] == "-":
                spy_nnnn.append(True_SPY_train[bbbb+4])
            else:
                spy_pppp.append(True_SPY_train[bbbb+4])
        elif True_SPY_train[bbbb] == True_SPY_train[bbbb+1] == True_SPY_train[bbbb+2] != True_SPY_train[bbbb+3]:
            if True_SPY_train[bbbb] == "-":
                spy_nnnp.append(True_SPY_train[bbbb+4])
            else:
                spy_pppn.append(True_SPY_train[bbbb+4])
        elif True_SPY_train[bbbb] == True_SPY_train[bbbb+1] != True_SPY_train[bbbb+2] == True_SPY_train[bbbb+3]:
            if True_SPY_train[bbbb] == "-":
                spy_nnpp.append(True_SPY_train[bbbb+4])
            else:
                spy_ppnn.append(True_SPY_train[bbbb+4]) 
        elif True_SPY_train[bbbb] == True_SPY_train[bbbb+1] == True_SPY_train[bbbb+3] != True_SPY_train[bbbb+2]:
            if True_SPY_train[bbbb] == "-":
                spy_nnpn.append(True_SPY_train[bbbb+4])
            else:
                spy_ppnp.append(True_SPY_train[bbbb+4]) 
        elif True_SPY_train[bbbb] == True_SPY_train[bbbb+2] == True_SPY_train[bbbb+3] != True_SPY_train[bbbb+1]:
            if True_SPY_train[bbbb] == "-":
                spy_npnn.append(True_SPY_train[bbbb+4])
            else:
                spy_pnpp.append(True_SPY_train[bbbb+4]) 
        elif True_SPY_train[bbbb] == True_SPY_train[bbbb+2] != True_SPY_train[bbbb+3] == True_SPY_train[bbbb+1]:
            if True_SPY_train[bbbb] == "-":
                spy_npnp.append(True_SPY_train[bbbb+4])
            else:
                spy_pnpn.append(True_SPY_train[bbbb+4])
        elif True_SPY_train[bbbb+1] == True_SPY_train[bbbb+2] == True_SPY_train[bbbb+3] != True_SPY_train[bbbb]:
            if True_SPY_train[bbbb] == "-":
                spy_nppp.append(True_SPY_train[bbbb+4])
            else:
                spy_pnnn.append(True_SPY_train[bbbb+4])
        elif True_SPY_train[bbbb] == True_SPY_train[bbbb+3] != True_SPY_train[bbbb+2] == True_SPY_train[bbbb+1]:
            if True_SPY_train[bbbb] == "-":
                spy_nppn.append(True_SPY_train[bbbb+4])
            else:
                spy_pnnp.append(True_SPY_train[bbbb+4])
        bbbb = bbbb+1

print("Probability of up day SPY pppp:", round((spy_pppp.count("+"))/len(spy_pppp),2))
print("Probability of up day SPY pppn:", round((spy_pppn.count("+"))/len(spy_pppn),2))
print("Probability of up day SPY ppnn:", round((spy_ppnn.count("+"))/len(spy_ppnn),2))
print("Probability of up day SPY ppnp:", round((spy_ppnp.count("+"))/len(spy_ppnp),2))
print("Probability of up day SPY pnpp:", round((spy_pnpp.count("+"))/len(spy_pnpp),2))
print("Probability of up day SPY pnpn:", round((spy_pnpn.count("+"))/len(spy_pnpn),2))
print("Probability of up day SPY pnnp:", round((spy_pnnp.count("+"))/len(spy_pnnp),2))
print("Probability of up day SPY pnnn:", round((spy_pnnn.count("+"))/len(spy_pnnn),2))

print("Probability of up day SPY nnnn:", round((spy_nnnn.count("+"))/len(spy_nnnn),2))
print("Probability of up day SPY nnnp:", round((spy_nnnp.count("+"))/len(spy_nnnp),2))
print("Probability of up day SPY nnpp:", round((spy_nnpp.count("+"))/len(spy_nnpp),2))
print("Probability of up day SPY nnpn:", round((spy_nnpn.count("+"))/len(spy_nnpn),2))
print("Probability of up day SPY nppp:", round((spy_nppp.count("+"))/len(spy_nppp),2))
print("Probability of up day SPY nppn:", round((spy_nppn.count("+"))/len(spy_nppn),2))
print("Probability of up day SPY npnp:", round((spy_npnp.count("+"))/len(spy_npnp),2))
print("Probability of up day SPY npnn:", round((spy_npnn.count("+"))/len(spy_npnn),2))



pppp_spy = "-"
pppn_spy = "+"
ppnn_spy = "+"
ppnp_spy = "-"
pnpp_spy = "+"
pnpn_spy = "+"
pnnp_spy = "+"
pnnn_spy = "+"
nnnn_spy = "+"
nnnp_spy = "+"
nnpp_spy = "-"
nnpn_spy = "-"
npnn_spy = "+"
nppp_spy = "-"
npnp_spy = "-"
nppn_spy = "-"


#Create list of labels for SPY W3

spy_w4 = []

#Checks prior true labels and assigns next label 
kk4 = 755
while kk4 < 1258:
    if SPY_true[kk4-4] == SPY_true[kk4-3] == SPY_true[kk4-2] == SPY_true[kk4-1]:
        if SPY_true[kk4-4] == "+":
            spy_w4.append(pppp_spy)
        else:
            spy_w4.append(nnnn_spy)
    elif SPY_true[kk4-4] == SPY_true[kk4-3] == SPY_true[kk4-2] != SPY_true[kk4-1]:
        if SPY_true[kk4-4] == "+":
            spy_w4.append(pppn_spy)
        else:
            spy_w4.append(nnnp_spy)
    elif SPY_true[kk4-4] == SPY_true[kk4-3] != SPY_true[kk4-2] == SPY_true[kk4-1]:
        if SPY_true[kk4-4] == "+":
            spy_w4.append(ppnn_spy)
        else:
            spy_w4.append(nnpp_spy)
    elif SPY_true[kk4-4] == SPY_true[kk4-3] == SPY_true[kk4-1] !=SPY_true[kk4-2]:
        if SPY_true[kk4-4] == "+":
            spy_w4.append(ppnp_spy)
        else:
            spy_w4.append(nnpn_spy)      
    elif SPY_true[kk4-3] == SPY_true[kk4-1] == SPY_true[kk4-2] != SPY_true[kk4-4] :
        if SPY_true[kk4-4] == "+":
            spy_w4.append(pnnn_spy)
        else:
            spy_w4.append(nppp_spy)  
    elif SPY_true[kk4-4] == SPY_true[kk4-2] !=SPY_true[kk4-1] == SPY_true[kk4-3] :
        if SPY_true[kk4-4] == "+":
            spy_w4.append(pnpn_spy)
        else:
            spy_w4.append(npnp_spy)  
    elif SPY_true[kk4-4] == SPY_true[kk4-2] == SPY_true[kk4-1] != SPY_true[kk4-3] :
        if SPY_true[kk4-4] == "+":
            spy_w4.append(pnpp_spy)
        else:
            spy_w4.append(npnn_spy) 
    elif SPY_true[kk4-4] == SPY_true[kk4-1] != SPY_true[kk4-2] == SPY_true[kk4-3] :
        if SPY_true[kk4-4] == "+":
            spy_w4.append(pnnp_spy)
        else:
            spy_w4.append(nppn_spy)   
    kk4 = kk4+1
    
# Compute accuracies, compare true label in year 4&5 to W4 computed label

print(len(spy_test_true))
print(len(spy_w4))


w4_tp_spy = 0
w4_fp_spy = 0
w4_tn_spy = 0
w4_fn_spy = 0

uu = 0
for e in spy_test_true:
    if spy_test_true[uu] == spy_w4[uu]:
        if spy_test_true[uu] =="+":
            w4_tp_spy = w4_tp_spy +1
        else:
            w4_tn_spy = w4_tn_spy+1
    else:
        if spy_w4[uu] == "+":
            w4_fp_spy = w4_fp_spy +1
        else:
            w4_fn_spy = w4_fn_spy+1
    uu = uu+1
    
print("SPY w4 TP:", w4_tp_spy, "FP", w4_fp_spy,"TN", w4_tn_spy , "FN", w4_fn_spy)


#Compute Ensemble 
FDX_ensemble = []

q = 0
for e in fdx_w2:
    if fdx_w2[q] == fdx_w3[q]:
        FDX_ensemble.append(fdx_w2[q])
    elif fdx_w2[q] == fdx_w4[q]:
        FDX_ensemble.append(fdx_w2[q])
    elif fdx_w3[q] == fdx_w4[q]:
        FDX_ensemble.append(fdx_w3[q])
    q = q+1

# Compute accuracies, compare true label in year 4&5 to Ensemble computed label

print(len(fdx_test_true))
print(len(FDX_ensemble))


ensemble_tp_fdx = 0
ensemble_fp_fdx = 0
ensemble_tn_fdx = 0
ensemble_fn_fdx = 0

uu = 0
for e in fdx_test_true:
    if fdx_test_true[uu] == FDX_ensemble[uu]:
        if fdx_test_true[uu] =="+":
            ensemble_tp_fdx = ensemble_tp_fdx +1
        else:
            ensemble_tn_fdx = ensemble_tn_fdx+1
    else:
        if FDX_ensemble[uu] == "+":
            ensemble_fp_fdx = ensemble_fp_fdx +1
        else:
            ensemble_fn_fdx = ensemble_fn_fdx+1
    uu = uu+1
    
print("FDX ensemble TP:", ensemble_tp_fdx, "FP", ensemble_fp_fdx,"TN", ensemble_tn_fdx , "FN", ensemble_fn_fdx)

        
#Compute Ensemble 
SPY_ensemble = []

q = 0
for e in fdx_w2:
    if fdx_w2[q] == fdx_w3[q]:
        SPY_ensemble.append(fdx_w2[q])
    elif fdx_w2[q] == fdx_w4[q]:
        SPY_ensemble.append(fdx_w2[q])
    elif fdx_w3[q] == fdx_w4[q]:
        SPY_ensemble.append(fdx_w3[q])
    q = q+1

        
# Compute accuracies, compare true label in year 4&5 to Ensemble computed label

print(len(spy_test_true))
print(len(SPY_ensemble))


ensemble_tp_spy = 0
ensemble_fp_spy = 0
ensemble_tn_spy = 0
ensemble_fn_spy = 0

uu = 0
for e in spy_test_true:
    if spy_test_true[uu] == SPY_ensemble[uu]:
        if spy_test_true[uu] =="+":
            ensemble_tp_spy = ensemble_tp_spy +1
        else:
            ensemble_tn_spy = ensemble_tn_spy+1
    else:
        if SPY_ensemble[uu] == "+":
            ensemble_fp_spy = ensemble_fp_spy +1
        else:
            ensemble_fn_spy = ensemble_fn_spy+1
    uu = uu+1
    
print("SPY ensemble TP:", ensemble_tp_spy, "FP", ensemble_fp_spy,"TN", ensemble_tn_spy , "FN", ensemble_fn_spy)


print('Question 5')

#Calculate buy and hold
 
buy_hold = []

value = 100
y=754
i = 0
while i < 503:
    value = value + (FDX_return[y]*value)
    buy_hold.append(value)
    y = y+1
    i = i+1

 


#Calculate buying on the strategy +
fdx_w3_growth =[]


value2 = 100
z = 754
for e in fdx_w3:
    if e == "+":
        value2 = value2 + (FDX_return[z]*value2)
        fdx_w3_growth.append(value2)
    else:
        fdx_w3_growth.append(value2)
    z = z+1
    
    

days = list(range(503))

x1 = days
y1 = buy_hold
# plotting the line 1 points 
plt.plot(x1, y1, label = "Buy and Hold")
# line 2 points
x2 = days
y2 = fdx_w3_growth
# plotting the line 2 points 
plt.plot(x2, y2, label = "FDX W3 strategy growth")
plt.xlabel('Days')
# Set the y axis label of the current axis.
plt.ylabel('Value')
# Set a title of the current axes.
plt.title('Buy & Hold vs Strategy Trading')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()




