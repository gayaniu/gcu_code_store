

import sys
import pyodbc
import pandas as pd
import json as json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
from markovchain import MarkovChain
np.set_printoptions(threshold=np.inf)

class Hmm(object):
    """docstring for ClassName"""
    def __init__(self):
        super(Hmm, self).__init__()       
        

    def hmmmResult(self, space):

        obs = ("normal", "cold", "dizzy")
        states = ("Healthy", "Fever")
        start_p = {"Healthy": 0.6, "Fever": 0.4}
        trans_p = { "Healthy": {"Healthy": 0.7, "Fever": 0.3},
                    "Fever": {"Healthy": 0.4, "Fever": 0.6}, }
        emit_p = {      "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
                        "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6}, }

        x = self.viterbi(obs,        states,        start_p,        trans_p,        emit_p)
        print(type(obs))
        print (x)


    def get_space_data(self, space):

        server = 'PEMIL' 
        database = 'HMM_HomeStates' 
        username = 'sa' 
        password = '123.123' 
        conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
        #cursor = conn.cursor()

        #cursor.execute("""EXEC [dbo].[SP_Remove_Anomalies] 1, 'kitchen' """)
        cmd_prod_executesp = """EXEC   [dbo].[SP_Get_Occupancy_Data_By_Space]  'kitchen' """
        df = pd.read_sql(cmd_prod_executesp, conn)

        print(df)   

        conn.close()
        return df

    def hmmResult_occupancy(self, sp):

        obs = self.get_space_data('kitchen')
    
        obs_1 = pd.DataFrame(obs['First_Label'])
    
        xy = obs_1.transform(lambda x: list(zip(x, obs_1['First_Label'])))
        

        tuples = [tuple(x) for x in obs_1.values]
        obs= tuple([tuple(obs_1['First_Label']) for col in obs_1])
        print(type(obs))

        obs = ('Short_Inactive', 'Active', 'Long_Inactive', 'Active', 'Long_Inactive', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Active', 'Long_Inactive', 'Active', 'Long_Inactive', 'Active', 'Long_Inactive', 'Active', 'Short_Inactive', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Active', 'Long_Inactive', 'Active', 'Long_Inactive', 'Active', 'Long_Inactive', 'Active', 'Long_Inactive', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Transit', 'Transit', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Transit', 'Transit', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Active', 'Long_Inactive', 'Active', 'Transit', 'Active', 'Short_Inactive', 'Active', 'Short_Inactive', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Transit', 'Transit', 'Transit', 'Transit', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Active', 'Short_Inactive', 'Active', 'Long_Inactive', 'Transit', 'Active', 'Short_Inactive', 'Active', 'Long_Inactive', 'Transit', 'Long_Inactive|TR', 'Long_Inactive|TR', 'Active', 'Short_Inactive', 'Active', 'Long_Inactive', 'Active')

        states = ('OCCUPIED_ACTIVE','OCCUPIED_INACTIVE','UNOCCUPIED','UNKNOWN')
        start_p = {'OCCUPIED_ACTIVE': 0.25, 'UNOCCUPIED': 0.25, 'OCCUPIED_INACTIVE':0.25, 'UNKNOWN':0.25}
        trans_p = { 'OCCUPIED_ACTIVE': {'OCCUPIED_ACTIVE':0.5,'UNOCCUPIED':0.2,'OCCUPIED_INACTIVE':0.2,'UNKNOWN':0.1},
                     'UNOCCUPIED': {'OCCUPIED_ACTIVE':0.10,'UNOCCUPIED':0.6,'OCCUPIED_INACTIVE':0.0,'UNKNOWN':0.3},
                    'OCCUPIED_INACTIVE': {'OCCUPIED_ACTIVE':0.35,'UNOCCUPIED':0.0,'OCCUPIED_INACTIVE':0.6,'UNKNOWN':0.05},
                    'UNKNOWN': {'OCCUPIED_ACTIVE':0.5,'UNOCCUPIED':0.3,'OCCUPIED_INACTIVE':0.0,'UNKNOWN':0.2} , }    


        emit_p = {  'OCCUPIED_ACTIVE': {'Active':0.6,'Short_Inactive':0.15,'Long_Inactive':0.1,'Long_Inactive|TR':0.05, 'Transit':0.10},
        'OCCUPIED_INACTIVE': {'Active':0.1,'Short_Inactive':0.15,'Long_Inactive':0.65,'Long_Inactive|TR':0.05, 'Transit':0.05}, #0.10 ,0.15, 0.65 ,0.05, 0.05
        'UNOCCUPIED': {'Active':0.05,'Short_Inactive':0.0,'Long_Inactive':0.25,'Long_Inactive|TR':0.7, 'Transit':0.0},     
        'UNKNOWN': {'Active':0.0,'Short_Inactive':0.0,'Long_Inactive':0.05,'Long_Inactive|TR':0.15, 'Transit':0.8}, }

        x = self.viterbi(obs, states, start_p, trans_p, emit_p)
        print(type(obs))
        print (x)


    def viterbi(self, obs, states, start_p, trans_p, emit_p):
        V = [{}]
        for st in states:
            V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob * emit_p[st][obs[t]]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        for line in self.dptable(V):
            print(line)

        opt = []
        max_prob = 0.0
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

    def dptable(self, V):
        # Print a table of steps from dictionary
        yield " ".join(("%12d" % i) for i in range(len(V)))
        for state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)



ex = Hmm()
#pydot.find_graphviz()
#print(pydot.find_graphviz.__doc__)
#print(ex.define_initial_states())
print(ex.hmmResult_occupancy('kitchen'))
#print(ex.get_space_data('kitchen')