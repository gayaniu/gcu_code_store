#import ujson
import collections
import dateutil.parser
from hmmstate import HMMState
#from hiddensequence import HiddenSequence
import os
import pyodbc
import numpy as np
import pandas as pd


#from decoder import Decoder
class HiddenMarkov:

    def __init__(self, space):
        self.space = space
        self.hmmStateCache = collections.deque([], NUM_HMMSTATES)
        self.modelFile = None
        self.transProbs ={
        'OCCUPIED_ACTIVE': {'OCCUPIED_ACTIVE':0.5,'UNOCCUPIED':0.2,'OCCUPIED_INACTIVE':0.2,'UNKNOWN':0.1},
        'UNOCCUPIED': {'OCCUPIED_ACTIVE':0.10,'UNOCCUPIED':0.6,'OCCUPIED_INACTIVE':0.0,'UNKNOWN':0.3},
        'OCCUPIED_INACTIVE': {'OCCUPIED_ACTIVE':0.35,'UNOCCUPIED':0.0,'OCCUPIED_INACTIVE':0.6,'UNKNOWN':0.05},
        'UNKNOWN': {'OCCUPIED_ACTIVE':0.5,'UNOCCUPIED':0.3,'OCCUPIED_INACTIVE':0.0,'UNKNOWN':0.2}
        }
       
        self.emissionProbs = {
        'OCCUPIED ACTIVE' : {'Transit':0.55, 'Short_Active' :0.05, 'Long_Active':0.3,'Short_Inactive':0.05,'Long_Inactive':0.05},
        'OCCUPIED INACTIVE' : {'Transit':0.05, 'Short_Active':0.0, 'Long_Active':0.0,'Short_Inactive':0.6,'Long_Inactive':0.35},
        'UNOCCUPIED' : {'Transit':0.05, 'Short_Active':0.0, 'Long_Active':0.0,'Short_Inactive':0.05,'Long_Inactive':0.9},
        'UNKNOWN' : {'Transit':0.8, 'Short_Active':0.0, 'Long_Active':0.0,'Short_Inactive':0.15,'Long_Inactive':0.05}
        }

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
    

    def addTohmmCache(self):
        global NUM_HMMSTATES
        x =  self.get_space_data ('kitchen')
        hstate = HMMState(dateutil.parser.parse(x['From']),x['First_Label'],int(x['Time_Gap_SS']))
        print(hstate)
        # x= msg.split(',')
        # hstate = HMMState(dateutil.parser.parse(x[1]),x[2],int(x[3]))
        # print("addTohmmCache "+hstate.toString())
        # self.hmmStateCache.append(hstate)




        # if len(self.hmmStateCache) >= NUM_HMMSTATES:
        #     l = list()
        #     l2 = list()
        #     for x in self.hmmStateCache:
        #         l.append(x.label)
        #         l2.append(x)
        #         print ("x in self.hmmStateCache: %s , %s , %s " % (str(x.fromTimestamp), x.label, str(x.duration)))
        #     l = [x.strip(' ') for x in l]
        #     opt = self.viterbi(l,occupancyStates, start_probability_occ, self.transProbs, self.emissionProbs, l2)
        #     maxpr = opt.probability
        #     hiddensts=opt.hmmstates
        #     state = hiddensts[len(hiddensts)-1]
        #     hmm_payload = {
        #         'model':       'hidden-markov-model',
        #         'space_id':  self.space.id,
        #         'occurred_at': str(state.fromTimestamp),
        #         'duration':    state.duration,
        #         'state':       state.hiddenstate,
        #         'state_probability': str(maxpr)}
        #     topic = "ai_model_results/occupancy/"+self.space.name+"/"
        #     print(topic)
      

    def viterbi(self,obs, states, start_p, trans_p, emit_p, obs2):
        global NUM_HMMSTATES
        try:

            V = [{}]
            for st in states:
                V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
                # Run Viterbi when t > 0
            for t in range(1, len(obs)):
                V.append({})
                for st in states:
                    max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
                    for prev_st in states:
                        if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                            max_prob = max_tr_prob * emit_p[st][obs[t]]

                            V[t][st] = {"prob": max_prob, "prev": prev_st}
                            break
            0
            for line in self.dptable(V):
                print (line)
            opt = []
            opt2 = []
            # The highest probability
            max_prob = max(value["prob"] for value in V[-1].values())
            previous = None
            # Get most probable state and its backtrack
            index = 0

            for st, data in V[-1].items():
                if data["prob"] == max_prob:
                    opt.append(st)
                    hd = HMMState(obs2[NUM_HMMSTATES-1].fromTimestamp,obs2[NUM_HMMSTATES-1].label,obs2[NUM_HMMSTATES-1].duration)
                    hd.hiddenstate = st
                    hd.probablity = max_prob
#                     print("V data "+str(data)+" st: "+st+"max : "+str(max_prob)+"  "+str(index)+" "+hd.toString())
                    opt2.append(hd)
                    index=index+1
                    previous = st
                    break

            # Follow the backtrack till the first observation

            index = len(V)
            for t in range(len(V) - 2, -1, -1):
                prevState = V[t + 1][previous]["prev"]
                opt.insert(0, V[t + 1][previous]["prev"])
                hd2 = HMMState(obs2[t].fromTimestamp,obs2[t].label,obs2[t].duration)
                hd2.hiddenstate = prevState

                opt2.insert(0, hd2)
                previous = V[t + 1][previous]["prev"]

            print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

            index = 0
            #hseq = HiddenSequence(opt2, max_prob)

            return opt2, max_prob

        except Exception as e:
            print("ERROR in viterbi")
            print(e.message)

    def dptable(self,V):
        # Print a table of steps from dictionary
        yield " ".join(("%12d" % i) for i in range(len(V)))
        for state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)




   
  

MODEL_FILE_EXTENSION = ".txt"
MODEL_FILE_FOLDER = "trained_models/"

occupancyStates = ('OCCUPIED_ACTIVE','UNOCCUPIED','OCCUPIED_INACTIVE','UNKNOWN')

start_probability_occ = {'OCCUPIED_ACTIVE': 0.25, 'UNOCCUPIED': 0.25, 'OCCUPIED_INACTIVE':0.25, 'UNKNOWN':0.25}
#for viterbi_r
labels = ['Transit', 'Short_Active', 'Long_Active','Short_Inactive','Long_Inactive']


NUM_HMMSTATES = 2 #at least 2
# hmmStateCache = collections.deque([], NUM_HMMSTATES)
ex = HiddenMarkov('kitchen')
ex.addTohmmCache()
