# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:52:20 2024

@author: usha6
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:01:10 2024

@author: pjdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read input files
network_data = pd.read_excel("~/Desktop/hydrologic project/Network_table_13.xlsx")
forcing_data = pd.read_csv("~/Desktop/hydrologic project/P_T_ET.txt", sep='\s+', parse_dates=['Date'])
observed_flow = pd.read_excel("~/Desktop/hydrologic project/Q.xlsx")

# Set date as index
forcing_data = forcing_data.set_index('Date')
observed_flow = observed_flow.set_index('Date')

# Set parameters -- those values that might change with each model run or 
# watershed
dtis = 3600  # sec, 1 hr
ndx = 1
p_W = 1.0  # 1 m wide for all model units
unitflow = 0.0001

#%% Preprocess network settings
# Add cumulative upstream area column to dataframe for calculating later
network_data['chuparea'] = np.zeros(len(network_data))

# Set up arrays to hold Muskingum-Cunge values for later processing
p_DX = np.zeros(len(network_data))
p_cc1 = np.zeros(len(network_data))
p_cc2 = np.zeros(len(network_data))
p_cc3 = np.zeros(len(network_data))
p_cc4 = np.zeros(len(network_data))

ch_DX = np.zeros(len(network_data))
ch_cc1 = np.zeros(len(network_data))
ch_cc2 = np.zeros(len(network_data))
ch_cc3 = np.zeros(len(network_data))
ch_cc4 = np.zeros(len(network_data))

# Algorithm to process network areas
for i, r in network_data.iterrows():
    # Calculate cumulative upstream area for each subcatchment that has upstream
    # subcatchments; note that this depends on the appropriate numbering of
    # subcatchments, as with flow
    if r['Numup'] == 2:
        network_data.loc[i, 'chuparea'] = (r['Lp_m'] * r['Lch_m'] * 2 / 1e6) + \
            network_data.loc[network_data['ModelID'] == r['upID1'], 'chuparea'].values[0] + \
            network_data.loc[network_data['ModelID'] == r['upID2'], 'chuparea'].values[0]
    else:
        # Calculate subcatchment area for those without upstream areas
        network_data.loc[i, 'chuparea'] = r['Lp_m'] * r['Lch_m'] * 2 / 1e6

    # Calculate the length of reach and plane sections if we want to deal with 
    # them in multiple steps
    ch_DX[i] = r['Lch_m'] / ndx  # m to km
    p_DX[i] = r['Lp_m'] / ndx

# Algorithm to calculate Muskingum Cunge parameters
for i,r in network_data.iterrows():
    
    # Define q_ref for both plane and channel with unit flow as described in 
    # assignment
    p_q_ref = ((r['Lp_m'] * 1) / 1000**2) * unitflow
    ch_q_ref = (r['chuparea']) * unitflow #make sure do not divide again cux you be dividig again
    nch= 0.03
    n_p= 0.30

    # Define channel variables for constant parameter Muskingum Cunge approach, 
    # and then Muskingum Cunge C parameters for channel
    C1 = (1.0 * r['Sch_mpm']**0.5) / nch
    y = (ch_q_ref / (C1 * r['wch_m']))**0.6
    Ax = r['wch_m'] * y
    celert = (5.0 / 3.0) * ch_q_ref / Ax
    tv = dtis / ch_DX[i]
    ch_C = celert * tv
    ch_D = (ch_q_ref / r['wch_m']) / (r['Sch_mpm'] * celert * ch_DX[i])
    cdenom = 1 + ch_C + ch_D
    ch_cc1[i] = (1 + ch_C - ch_D) / cdenom
    ch_cc2[i] = (-1 + ch_C + ch_D) / cdenom
    ch_cc3[i] = (1 - ch_C + ch_D) / cdenom
    ch_cc4[i] = 2.0 * ch_C / cdenom

    # Define hillslope variables for constant parameter Muskingum Cunge approach, 
    # and then Muskingum Cunge C parameters for hillslope
    C1 = (1.0 * r['Sp_mpm']**0.5) / n_p
    y = (p_q_ref / (C1 * p_W))**0.6
    Ax = p_W * y
    celert = (5.0 / 3.0) * p_q_ref / Ax
    tv = dtis / p_DX[i]
    p_C = celert * tv
    p_D = (p_q_ref / p_W) / (r['Sp_mpm'] * celert * p_DX[i])
    cdenom = 1 + p_C + p_D
    p_cc1[i] = (1 + p_C - p_D) / cdenom
    p_cc2[i] = (-1 + p_C + p_D) / cdenom
    p_cc3[i] = (1 - p_C + p_D) / cdenom
    p_cc4[i] = 2.0 * p_C / cdenom

# Set number of subcatchments as a variable for later use
nsc = len(network_data)


#%% Set parameters -- those values that might change with each watershed;
# in this case, these are parameters we can't easily measure so want to calibrate

# Infiltration and groundwater parameters
ai = 0.1  # cm/hr
bR = 1
aR = 0.004  # cm/hr
aGW = 0.0001  # cm/hr

# Snow parameters
Tsnow = 1.50  # degree C
m1 = 0.003  # cm/hr/degree C
m2 = 2 * m1

# Soil moisture storage parameters
minSS = 0
maxSS = 50

#%% Define important functions                   
        
def calcKGE(obs, mod):
    # Take in two time series of the same size, obs and mod, 
    # and calculate the KGE
    
    # obs = data3['Qcms'].values
    # mod = DailyQ['hourlyQ'].values
    # First, calculate the components
    l = len(obs)
    muObs, muMod = np.mean(obs), np.mean(mod) # mean
    sigObs = np.sqrt(np.sum((obs-muObs)**2)/l)
    sigMod = np.sqrt(np.sum((mod-muMod)**2)/l)
    cvObs, cvMod = sigObs/muObs, sigMod/muMod
    cov = np.sum((obs-muObs)*(mod-muMod),axis=0)/l

    # Then calculate the three terms in the KGE
    r = cov/(sigObs*sigMod)
    beta = muMod/muObs
    gamma = cvMod/cvObs
    # Then calculate and return KGE
    kge = 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
    return kge 

#%% Initialize variables -- sest up arrays that we will need to store data for 
# the flow calculations

nTs = len(forcing_data) # number of timesteps

# Subcatchment network information
numup = network_data['Numup'].to_numpy()
upID1 = network_data['upID1'].to_numpy(dtype='int') - 1
upID2 = network_data['upID2'].to_numpy(dtype='int') - 1

# State variables for soil moisture, snow, and groundwater
SM = np.zeros(nTs + 1)
Snow = np.zeros(nTs + 1)
GW = np.zeros(nTs + 1)

# Arrays to store channel and hillslope flow
ch_Q = [[0] for _ in range(nsc)]
p_Q = [[0] for _ in range(nsc)]

# Initialize all previous and upslope flows as 0 to start the model
p_q_in = p_q_in_old = p_q_out_old = p_q_out = 0
ch_q_in = ch_q_in_old = ch_q_out_old = ch_q_out = 0

# Define precipitation, temperature, and ET vectors as the observations and 
# calculations with a 0 on the front end for ease of indexing below 
precip = np.zeros(nTs+1)
precip[1:] = forcing_data['precipitation'].values
temperature = np.zeros(nTs+1)
temperature[1:] = forcing_data['Tair_C']
evaptrans = np.zeros(nTs+1)
evaptrans[1:] = forcing_data['Hourly_ET'].values



#%% Calculate time series

# loop over time, starting at 1 to allow for 0 initial values previously defined
for t in np.arange(1,nTs+1):

    # Load this timestep's forcings
    Tair = temperature[t]
    et = evaptrans[t]
    # If Tair <= Tsnow, change precip to snow
    p, pSnow = precip[t]*(Tair > Tsnow), precip[t]*(Tair <= Tsnow)

    ### Soil moisture
    # Calculate infiltration rate
    # if Snow[t - 1] > 0, snow on the ground so no infiltration
    irate = min(ai + ai * (1 - (SM[t-1] / maxSS)),p) * (Snow[t - 1] <= 0)

    # Calculate GW recharge rate
    Rrate = aR * (SM[t-1] / maxSS)**bR

    # Update soil moisture state variable
    SM[t] = SM[t-1] + irate - et - Rrate  # cm/hr to cm because DT is 1 hr
    
    # Rules to correct rates if soil moisture ends up too low or too high
    # If soil moisture is too low, first reduce ET and then reduce recharge
    # 
    if SM[t] < minSS:
        diffval = minSS - SM[t]
        if diffval <= et:
            et = diffval
        else: 
            Rrate = max(Rrate - (diffval - et),0)
            et = 0            
        SM[t] = minSS

    # If soil  moisutre is above maximum, reduce infiltration to keep it
    # at maximum
    if SM[t] > maxSS:
        irate -= SM[t] - maxSS
        SM[t] = maxSS

    # Groundwater
    # calcluate GW flow to the river
    GWrate = aGW * GW[t - 1]
    # Update groundwater state variable; cm/hr to cm because DT is 1 hr
    GW[t] = GW[t - 1] + Rrate - GWrate + (GW[t - 1] + Rrate)*(GWrate > GW[t - 1] + Rrate)   

    # Snow
    # Calculate effective melt rate based on precipitation & temperature
    mEffective = m2 * (p > 0) + m1 * ((pSnow == 0) & (p == 0))
    Mrate = mEffective * max(Tair - Tsnow,0)
    
    # Define snow change, correct the melt rate if it takes snow state variable
    # below 0, and then update snow state variable
    dSnow = pSnow - Mrate
    if (Snow[t-1] + dSnow) < 0: Mrate += Snow[t-1] + dSnow
    Snow[t] = max(Snow[t-1] + dSnow,0)


    # Surface runoff rate
    pe = p - irate + Mrate

    # Surface runoff rate in units for hillslope routing
    pe = pe / 100 / 3600  # cm/hr to m3/s

    for n in range(len(network_data)):
        # print(i,n)
    
        # Determine QL for plane
        p_QL = pe * 1 * p_DX[n]  # m3/s for 1 meter unit plane
        # Then determine three needed Q's for first iteration
        # if data2.iloc[i].name > t0:   p_q_out_old = p_Q[i,n]
        # if i > 0: 
        p_q_out_old = p_Q[n][t-1]
        # Then determine unit plane response
        p_q_out = p_cc3[n] * p_q_out_old + p_cc4[n] * p_QL

        # Determine QL for channel
        Pss = GWrate*(1/100)*(1/3600)*1*p_DX[n] #cm/hr to m3/s/m
        ch_QL = (p_q_out + Pss) * 2 * ch_DX[n]  # m3/s per unit plane times 2 planes
        # Then determine three needed Q's for first iteration
        # if data2.iloc[i].name > t0: 
        # if i > 0: 
        ch_q_out_old = ch_Q[n][t-1]
        ch_q_in_old = ch_q_in = 0
        if numup[n] > 0:
            ch_q_in_old = ch_Q[upID1[n]][t-1] + ch_Q[upID2[n]][t-1]
            ch_q_in = ch_Q[upID1[n]][t] + ch_Q[upID2[n]][t]
            
        # Then determine channel response
        ch_q_out =  ch_cc1[n] * ch_q_in_old + ch_cc2[n] * ch_q_in + \
                    ch_cc3[n] * ch_q_out_old + ch_cc4[n] * ch_QL
        
        # Append both plane and channel calculations to their lists
        ch_Q[n].append(max([ch_q_out,0]))            
        p_Q[n].append(p_q_out)
            

# Now sum up to get the daily 
qDF = pd.DataFrame({'hourly Q':ch_Q[n][1:],'hourly SM':SM[1:],
                    'hourly GW':GW[1:],'hourly snow':Snow[1:]},
                   index=forcing_data.index)

dailyVals = qDF.resample('d').mean()
dailyVals = dailyVals.rename(columns={'hourly Q':'Daily Q','hourly SM':'Daily SM',
                              'hourly GW':'Daily GW','hourly snow':'Daily snow'})

#%% Plot and summarize statistics

print(np.sum(observed_flow['Qcms']))
print(np.sum(dailyVals['Daily Q']))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dailyVals['Daily SM'][1500:2190], color='red', label='SM')
plt.plot(dailyVals['Daily GW'][1500:2190], color='blue', label='GW')
plt.plot(dailyVals['Daily snow'][1500:2190], color='green', label='Snow')
plt.xlabel("Time")
plt.ylabel("(cm)")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dailyVals['Daily Q'][1500:2190],label='Model')
plt.plot(observed_flow['Qcms'][1500:2190],label='Obs')

# plt.xlim(1500, 2190)
# plt.ylim(1, 100)
plt.xlabel("Time")
plt.ylabel("(m3/s)")
plt.legend(loc='upper right')
plt.show()

# Save results
dailyVals.to_csv("~/temp/cive7260/finalProject/Model_Q_python.csv", index=False)