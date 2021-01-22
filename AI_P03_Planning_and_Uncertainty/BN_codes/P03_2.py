
#VariableElimination.inference([B,E,A,J,M], ['A'], ['B', 'E', 'J','M'], {})

#VariableElimination.inference([B,E,A,J,M], ['B'], ['E','A'], {'J':1,'M':0})

import itertools


VarRange = {
'PatientAge':['0-30','31-65','65+'],
'CTScanResult':['Ischemic Stroke','Hemmorraghic Stroke'],
'MRIScanResult': ['Ischemic Stroke','Hemmorraghic Stroke'],
'StrokeType': ['Ischemic Stroke','Hemmorraghic Stroke', 'Stroke Mimic'],
'Anticoagulants': ['Used','Not used'],
'Mortality':['True', 'False'],
'Disability': ['Negligible', 'Moderate', 'Severe']
}

def replace(list,fromv,tov):
    ans = []
    for i in list:
        if(i==fromv):
            ans.append(tov)
        else:
            ans.append(i)
    return ans

class VariableElimination:
    @staticmethod
    def inference(factorList, queryVariables,
    orderedListOfHiddenVariables, evidenceList):
        for key, value in evidenceList.items():
             new_factor_list = []
             for factor in factorList:
                 if(key in factor.varList):
                     newnode = factor.restrict(key,value)
                     new_factor_list.append(newnode)
                 else:
                     new_factor_list.append(factor)

             factorList = new_factor_list


        for var in orderedListOfHiddenVariables:
            #Your code here
            mulNode = None
            new_factor_list =[]

            for factor in factorList:
                if(var in factor.varList):
                    new_factor_list.append(factor)
            res = new_factor_list[0]
            factorList.remove(res)
            for factor in new_factor_list[1:]:
                res = res.multiply(factor)
                factorList.remove(factor)
            res = res.sumout(var)
            factorList.append(res)
            #print ("RESULT:")
            #for factor in factorList:
            #    factor.printInf()

        #print("result")
        #for factor in factorList:
        #    factor.printInf()
        new_factor_list = []
        for factor in factorList:
            if factor.varList != []:
                new_factor_list.append(factor)
        factorList = new_factor_list

        res = factorList[0]

        for factor in factorList[1:]:
            res = res.multiply(factor)
        total = sum(res.cpt.values())
        res.cpt = {k: v/total for k, v in res.cpt.items()}
        res.printInf()


    @staticmethod
    def printFactors(factorList):
        for factor in factorList:
            factor.printInf()
class Util:
    @staticmethod
    def to_binary(num, len):
        return format(num, '0' + str(len) + 'b')
class Node:
    def __init__(self, name, var_list):
        self.name = name
        self.varList = var_list
        self.cpt = {}
    def setCpt(self, cpt):
        self.cpt = cpt
    def printInf(self):
        print("Name = " + self.name)
        print(" vars " + str(self.varList) )
        for key in self.cpt:
            print("   key: " + str(key)+ " val : " + str(self.cpt[key]) )
        print("")

    def multiply(self, factor):
        """function that multiplies with another factor"""
        #Your code here
        new_cpt = {}
        newList = []
        samenode = ""
        index1 = 0
        index2 = 0
        for i,var in enumerate(self.varList):
            if var not in factor.varList:
                newList.append(var)
            else:
                index1 = i
                newList.append(var)
        for i,var in enumerate(factor.varList):
            if var not in newList:
                newList.append(var)
            else:
                index2 = i
        #print(self.cpt)
        #print(factor.cpt)
        #print ()
        #print(index1,index2)
        for tup1, prob1 in self.cpt.items():
            for tup2, prob2 in factor.cpt.items():
                newl = []
                if(tup2[index2] == tup1[index1]):
                    for i in tup1:
                        newl.append(i)
                    for i,ch in enumerate(tup2):
                        if(i!=index2):
                            newl.append(ch)
                    if(tuple(newl) not in new_cpt.keys() ):
                        new_cpt[tuple(newl)] = prob1 * prob2
        #print(new_cpt)
        #new_cpt =
        tempstr = ""
        new_node = Node("f" + str(newList), newList)
        new_node.setCpt(new_cpt)
        return new_node

    def sumout(self, variable):

        #dict structure: {(s1,s2..):prob1,(s1,s2..):prob2}

        """function that sums out a variable given a factor"""
        #Your code here
        new_var_list = []
        indexnum = 0
        for i,var in enumerate(self.varList):
            if(var != variable):
                new_var_list.append(var)
            else:
                indexnum = i
        new_cpt = {}
        tuplen = len(self.varList)
        tups = ()
        for ttup in self.cpt.keys() :
            tot  = 0
            listVar = []
            listt = []
            for i,s in enumerate(ttup):
                if(indexnum == i):
                    for j in VarRange[variable]:
                        listVar.append(j)
                    listt.append('')
                else:
                    listt.append(s)
            for var in listVar:
                newl = replace(listt,'',var)
                tot += self.cpt[tuple(newl)]
            newname = []
            for name in listt:
                if(name!=''):
                    newname.append(name)
            new_cpt[tuple(newname)] = tot

        #A.setCpt({'111': 0.95, '011': 0.05, '110': 0.94, '010': 0.06,
        #          '101': 0.29, '001': 0.71, '100': 0.001, '000': 0.999})

        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        return new_node
    def restrict(self, variable, value):
        """function that restricts a variable to some value
        in a given factor"""
        #Your code here
        new_var_list =[]
        indexnum = 0
        for i,var in enumerate(self.varList):
            if(var != variable):
                new_var_list.append(var)
            else:
                indexnum = i

        new_cpt ={}

        for tup in self.cpt.keys():
            flag = 0
            newt = []
            for i,s in enumerate(tup):
                if indexnum == i:
                    if s == value:
                        flag = 1
                else:
                    newt.append(s)
            newt = tuple(newt)
            if(newt not in new_cpt.keys() and flag ):
                new_cpt[newt] =self.cpt[tup]

        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        return new_node
# create nodes for Bayes Net
B = Node("B", ["B"])

E = Node("E", ["E"])
A = Node("A", ["A", "B","E"])
J = Node("J", ["J", "A"])
M = Node("M", ["M", "A"])

PatientAge =  Node("PatientAge",["PatientAge"])
CTScanResult = Node("CTScanResult",["CTScanResult"])
MRIScanResult = Node("MRIScanResult",['MRIScanResult'])
Anticoagulants = Node("Anticoagulants",['Anticoagulants'])
StrokeType = Node('StrokeType',['CTScanResult','MRIScanResult','StrokeType'])
Mortality = Node('Mortality',['StrokeType','Anticoagulants','Mortality'])
Disability = Node('Disability',['StrokeType','PatientAge','Disability'])

PatientAge.setCpt({ ('0-30',):0.10,('31-65',):0.30,('65+',):0.60})

CTScanResult.setCpt({ ('Ischemic Stroke',) :0.7, ('Hemmorraghic Stroke',):0.3})
MRIScanResult.setCpt({ ('Ischemic Stroke',):0.7, ('Hemmorraghic Stroke',):0.3})
Anticoagulants.setCpt({ ('Used',):0.5, ('Not used',):0.5})


Scpt = {}
listS =[]
StrokeTypeList = [
['Ischemic Stroke','Ischemic Stroke','Ischemic Stroke',0.8],
['Ischemic Stroke','Hemmorraghic Stroke','Ischemic Stroke',0.5],
[ 'Hemmorraghic Stroke','Ischemic Stroke','Ischemic Stroke',0.5],
[ 'Hemmorraghic Stroke','Hemmorraghic Stroke','Ischemic Stroke',0],

['Ischemic Stroke','Ischemic Stroke','Hemmorraghic Stroke',0],
['Ischemic Stroke','Hemmorraghic Stroke','Hemmorraghic Stroke',0.4],
[ 'Hemmorraghic Stroke','Ischemic Stroke','Hemmorraghic Stroke',0.4],
[ 'Hemmorraghic Stroke','Hemmorraghic Stroke','Hemmorraghic Stroke',0.9],

['Ischemic Stroke','Ischemic Stroke','Stroke Mimic',0.2],
['Ischemic Stroke','Hemmorraghic Stroke','Stroke Mimic',0.1],
[ 'Hemmorraghic Stroke','Ischemic Stroke','Stroke Mimic',0.1],
[ 'Hemmorraghic Stroke','Hemmorraghic Stroke','Stroke Mimic',0.1],
]
for list in StrokeTypeList:
    tup = (list[0],list[1],list[2])
    Scpt[tup] = list[3]
StrokeType.setCpt(Scpt)


### ******test*****
#StrokeType.sumout('CTScanResult')
#newnode = StrokeType.restrict('CTScanResult','Ischemic Stroke')

Mcpt = {}
listM =[
['Ischemic Stroke', 'Used', 'False',0.28],
['Hemmorraghic Stroke', 'Used', 'False',0.99],
['Stroke Mimic', 'Used', 'False',0.1],
['Ischemic Stroke','Not used', 'False',0.56],
['Hemmorraghic Stroke', 'Not used', 'False',0.58],
['Stroke Mimic', 'Not used', 'False',0.05],

['Ischemic Stroke',  'Used' ,'True',0.72],
['Hemmorraghic Stroke', 'Used', 'True',0.01],
['Stroke Mimic', 'Used', 'True',0.9],
['Ischemic Stroke',  'Not used' ,'True',0.44],
['Hemmorraghic Stroke', 'Not used', 'True',0.42 ],
['Stroke Mimic', 'Not used', 'True',0.95]
]
for list in listM:
    tup = (list[0],list[1],list[2])
    Mcpt[tup] = list[3]
Mortality.setCpt(Mcpt)

Dcpt= {}
listD = [
['Ischemic Stroke',   '0-30','Negligible', 0.80],
['Hemmorraghic Stroke', '0-30','Negligible', 0.70],
['Stroke Mimic',        '0-30', 'Negligible',0.9],
['Ischemic Stroke',     '31-65','Negligible', 0.60],
['Hemmorraghic Stroke', '31-65','Negligible', 0.50],
['Stroke Mimic',        '31-65', 'Negligible',0.4],
['Ischemic Stroke',     '65+'  , 'Negligible',0.30],
['Hemmorraghic Stroke', '65+'  , 'Negligible',0.20],
['Stroke Mimic',        '65+'  , 'Negligible',0.1],

['Ischemic Stroke',     '0-30' ,'Moderate',0.1],
['Hemmorraghic Stroke', '0-30' ,'Moderate',0.2],
['Stroke Mimic',        '0-30' ,'Moderate',0.05],
['Ischemic Stroke',     '31-65','Moderate',0.3],
['Hemmorraghic Stroke', '31-65','Moderate',0.4],
['Stroke Mimic',        '31-65','Moderate',0.3],
['Ischemic Stroke',     '65+'  ,'Moderate',0.4],
['Hemmorraghic Stroke', '65+'  ,'Moderate',0.2],
['Stroke Mimic',        '65+'  ,'Moderate',0.1],

['Ischemic Stroke',     '0-30' ,'Severe',0.1],
['Hemmorraghic Stroke', '0-30' ,'Severe',0.1],
['Stroke Mimic',        '0-30' ,'Severe',0.05],
['Ischemic Stroke',     '31-65','Severe',0.1],
['Hemmorraghic Stroke', '31-65','Severe',0.1],
['Stroke Mimic',        '31-65','Severe',0.3],
['Ischemic Stroke',     '65+'  ,'Severe',0.3],
['Hemmorraghic Stroke', '65+'  ,'Severe',0.6],
['Stroke Mimic',        '65+'  ,'Severe',0.8]
]
for list in listD:
    tup = list[0],list[1],list[2]
    Dcpt[tup] = list[3]
Disability.setCpt(Dcpt)


#P1
VariableElimination.inference([PatientAge,CTScanResult,MRIScanResult,Anticoagulants,StrokeType,\
                              Mortality,Disability],['Mortality','CTScanResult'],\
                              ['MRIScanResult','Anticoagulants',\
                              'StrokeType','Disability'],{'PatientAge':'31-65'})

#P2
VariableElimination.inference([PatientAge,CTScanResult,MRIScanResult,Anticoagulants,StrokeType,\
                               Mortality,Disability],['Disability','CTScanResult'],\
                              ['Anticoagulants','StrokeType','Mortality'],\
                              {'PatientAge':'65+','MRIScanResult':'Hemmorraghic Stroke'})

#P3 written in the form of deleting unrelative variables
VariableElimination.inference([PatientAge,CTScanResult,MRIScanResult,StrokeType],
                              ['StrokeType'],\
                              [],\
                              {'PatientAge':'65+','CTScanResult':'Hemmorraghic Stroke','MRIScanResult':'Ischemic Stroke'})

#P3 no deleting, the result is the same
VariableElimination.inference([PatientAge,CTScanResult,MRIScanResult,Anticoagulants,StrokeType,\
                               Mortality,Disability],['StrokeType'],\
                              ['Anticoagulants','Mortality','Disability'],\
                              {'PatientAge':'65+','CTScanResult':'Hemmorraghic Stroke','MRIScanResult':'Ischemic Stroke'})
#P4
VariableElimination.inference([PatientAge,CTScanResult,MRIScanResult,Anticoagulants,StrokeType,\
                              Mortality,Disability],['Anticoagulants'],\
                              ['CTScanResult','MRIScanResult','StrokeType','Mortality','Disability'],\
                              {'PatientAge':'31-65'})
#P5
VariableElimination.inference([PatientAge,CTScanResult,MRIScanResult,Anticoagulants,StrokeType,\
                              Mortality,Disability],['Disability'],\
                              ['PatientAge','CTScanResult','MRIScanResult','Anticoagulants',\
                               'StrokeType','Mortality'],{})
