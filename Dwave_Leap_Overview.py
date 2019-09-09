#Boolean AND Gate
---------------------------------------------------------
#Example solves a simple problem of a Boolean AND gate
#Programming the QPU consists of two inputs:
    #Qubit bias weights
        #control the degree to which qubit tends to a particular state
    #Qubit coupling strengths
        #control the degree to which the two qubits tend to the same state

#Biases and couplings define an energy landscape and the QPU seeks:
    #the minimum energy of that landscape

#Formulation = Define the problem such that:
    #desired outcomes have low energy values
    #undesired outcomes have high energy values
    #QPU solves the proble by finding the low energy states

#The formulation is called an objective function
    #Corresponds to the Ising model used in statistical mechanics

#Penalty Function
    #represents the AND gate
    #for assignments of variables matching valid states of the gate
    #the function evaluates at a lower value than assignments of an invalid gate
    #D-wave system minimizes the BQM based on the penalty function
        #Finds those assignments of variables matching the valid gate states

#QUBO coefgicients for AND gate

Q = {('x1', 'x2'): 1, ('x1', 'z'): -2, ('x2', 'z'): -2, ('z', 'z'): 3}

-------------------------------------------------------------
#Solve the Problem by Sampling: Automated Minor-Embedding
-------------------------------------------------------------
#Set up the D-wave sampler

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

sampler = DWaveSampler(endpoint='https://URL_to_my_D-Wave_system/',
                       token='ABC-123456789012345678901234567890',
                       solver='My_D-Wave_Solver')

#Ask for 50000 samples

response = sampler_embedded.sample_qubo(Q, num_reads=5000)
for datum in response.data(['sample', 'energy', 'num_occurances',]):
    print(datum.sample, "Energy:", datum.energy, "Occurrences:", datum.num_occurrences)

#All the returned sampes represent valid value assignments for an AND gate,
    #and minimize (are low energy states of the BQM)
-----------------------------------------------------------
#Solve the Problem by Sampling: Non-automated Minor-Embedding
------------------------------------------------------------

#NOT Gates may be represented by a BQM in QUBO with the following coefficients

Q_not = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}

#Minor embedding maps the two problem variables x and z to indexed qubits

#Select the first node and print adjacent nodes, coupled qubits
print(sampler.adjacency[sampler.nodelist[0])

#Manually minor-embed the problem with FixedEmbeddingComposite

from dwave.system.composites import FixedEmbeddingComposite
sampler_embedded = FixedEmbeddingComposite(sampler, {'x':[0], 'z':[4]})
print(sampler_embedded.adjacency)

#Ask for 500 samples

response = sampler_embedded.sample_qubo(Q_not, num_reads=5000)
for datum in response.date(['sample', 'energy', 'num_occurences']):
      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ",
            datum.num_occurrences)
---------------------------------------------------------------------
#Minor-Embedding an AND Gate
---------------------------------------------------------------------
      
#3 Qubits cannot be connected in a closed loop, but 4 can be connected in a closed loop
#Manual minor-embedding using FixedEmbeddingComposite()
from dwave.system.composites import FixedEmbeddingComposite
embedding = {'x1': {1}, 'x2': {5}, 'z': {0, 4}}
sampler_embedded = FixedEmbeddingComposite(sampler, embedding)
print(sampler_embedded.adjaceny)

#We ask for 5000 samples
Q = {('x1', 'x2'): 1, ('x1', 'z'): -2, ('x2', 'z'): -2, ('z', 'z'): 3}
response = sampler_embedded.sample_qubo(Q, num_reads=5000)
for datum in response.data(['sample', 'energy', 'num_ocurrences']):
      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurences)

#Weaken the chain strength of the coupler between qubits 0 and 4
#print the range of values available
#The default chain strength is 2
#Loweing it to 0.25 the two qubits are not as strongly correlated
#The result is many returned samplers represent invalid states for an AND gate

print(sampler.properties['extended_j_range'])

sampler_embedded = FixedEmbeddingComposite(sampler, embedding)
response = sampler_embedded.sample_qubo(Q, num_reads=5000, chain_strength=0.25)
for datum in response.data(['sample', 'energy', 'num_occurrences']):
      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurences)


#Multiple-Gate Circuit
#This example solves a logic circuit problem.
#demonstrating using Ocean tools to solve a problem on a D-Wave System.

#This example demonstrates two formulations of constraints from the logic gates
#Single comprehensive constraint

import dwavebinarycsp

def logic_circuit(a, b, c, d, z):
      not1 = not b
      or2 = b or c
      and3 = a and not1
      or4 = or2 or d
      and5 = and3 and or4
      not6 = not or4
      or7 = and5 or not6
      return(z == or7)

csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
csp.add_constraint(logic_circuit, ['a', 'b', 'c', 'd', 'z'])

#Multiple Constraints

import dwavebinarycsp
import dwavebinarycsp.factories.constraint.gates as gates
import operator

csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
csp.add_constraint(operator.ne, ['b', 'not1']) #add NOT 1 gate
csp.add_constraint(gates.or_gate(['b', 'c', 'or2'])) #add OR 2 gate
csp.add_constraint(gates.and_gate(['a', 'not1', 'and3'])) # add AND 3 gate
      
csp.add_constraint(gates.or_gate(['d', 'or2', 'or4'])) # add OR 4 gate
csp.add_constraint(gates.and_gate(['and3', 'or4', 'and5'])) # add AND 5 gate
      
csp.add_constraint(operator.ne, ['or4', 'not6']) # add NOT 6 gate
csp.add_constraint(gates.or_gate(['and5', 'not6', 'z']) # add OR 7 gate

# Convert the binary constraint satisfaction problem to a binary quadratic model
bqm = dwavebinarycsp.stitch(csp)
                   
---------
#The next code sets up the D-Wave sampler
---------

from dwave.system.samplers import DWaveSampler
from dwave.systems.composites import Embedding Composite

sampler = EmbeddingComposite(DWaveSampler(endpoint= 'path', token= 'token', solver='SolverName'))

#Next we as for 1000 samples and separate those that satisfy the CSP from those that fail to do so.

response = sampler.sample(bqm, num_reads=1000)

# Check how many solutions meet the constraints (are valid)
valid, invalid, data = 0,0, []
for datum in response.data(['sample', 'energy', 'num_occurences']):
    if (csp.check(datum.sample)):
        valid = valid+datum.num_occurences
        for i in range(datum.num_occurences):
            data.append((datum.sample, datum.energy, '1'))
    else:
        invalid = invalid+datum.num_occurences
        for i in range(datum.num_occurences):
            data.append((datum.sample, datum.energy, '0'))
                   
print(valid, invalid)
            
----
#Verify Solution
----

print(next(response.sample()))

----

#Plot the energies for valid and invalid samples

import matplotlib.pyplot as plt
plt.ion()
plt.scatter(range(len(data)), [x[1] for x in data], c=('y' if (x[2] == '1')
            else 'r' for x in data],marker='.')
plt.xlabel('Sample')
plt.ylabel('Energy')

#datum
                   
for datum in response.data(['sample', 'energy', 'num_occurences', 'chain_break_fraction']):
    print(datum)

#sample

Sample(sample={'a':1, 'c': 1, 'b': 0, 'not1': 1, 'd': 1, 'or4': 1, 'or2': 1, 'not6': 1, 'and5': 1, 'z': 1, 'and3': 1}, energy=-7.5, num_occurences=1, chain_break_fraction=0.1818181818)

Constraint.from_configurations(frozenset([(1, 0, 0), (0,1,0), (0,0,0), (1,1,1)]), ('a', 'not1', 'and3'), Vartype.BINARY, name='AND')

    
