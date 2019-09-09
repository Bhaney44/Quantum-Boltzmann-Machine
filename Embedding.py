
# @bhaney44

# Solving an Exactly-One-True problem

# Defining the problem

# Converting to a QUBO problem


# Here in our QUBO:
    # Each variable is a node weighted with its linear coefficient
    # Each quadratic term is an edge with a strength between the nodes

# Then, our QUBO can be represented by the following undirected graph.


# Putting the QUBO on Chimera

# Chimera structure
# The Chimera architecture comprises sets of connected unit cells, each with four horizontal qubits connected to four vertical qubits via couplers.
# Unit cells are tiled vertically and horizontally with adjacent qubits connected, creating a lattice of sparsely connected qubits.
# The D-Wave 2000Q QPU supports a C16 Chimera graph: its 2048 qubits are logically mapped into a 16 x 16 matrix of unit cells of 8 qubits.

# In a D-Wave QPU, the set of qubits and couplers that are available for computation is known as the working graph.
# The yield (working qubits) of a __working graph__ is typically less than the total number of qubits and couplers that are fabricated and physically present in the QPU.

# More descriptions about Chimera graph can be found [here](https://cloud.dwavesys.com/qubist/docs/sys_intro_getting_started/chimera/).

# Minor-embedding

# Our question is: How do we put this QUBO on D-Wave quantum computer?
    # A technique that can achieve this is called _minor-embedding_.
    #A minor-embedding of a graph in a quantum hardware graph is a subgraph such that can be obtained from it by contracting edges
        #[Choi2010](https://arxiv.org/pdf/1001.3116.pdf).
    #A minor-embedding of graph requires creating virtual qubit out of multiple physical qubits using a concept called _chaining_.

# Embedding a three-sided graph
# If you look at the Chimera graph, there are no triangles, all of the cycles are of at least length 4. 

# Using the concept of minor-embedding, we can make use of the cycle of length 4 to _construct_ a complete graph.


# Here are some embedding rules that we should keep in mind:

    #A logical qubit can be mapped to n physical qubits as long as the physical qubits form a connected set, called a chain.
    #For each physical coupler connecting two physical qubits in the same chain, include QUBO terms to cause the two physical qubits to align.
    #When a logical qubit is mapped to an n-qubit chain, divide the bias (weight) for the logical qubit by n and apply that bias to each qubit in the chain.
    #Split the coupling strength between two logical qubits over all available physical couplers connecting the chains corresponding to the logical qubits.

# Submitting problems to the quantum computer

from __future__ import print_function
from dwave.system.samplers import DWaveSampler

# create sampler.
# Some accounts need to replace this line with the next:
sampler = DWaveSampler(solver={'qpu': True})

# sampler = DWaveSampler(token = 'my_token', solver='solver_name')
# define the QUBO problem
w = 3.
Q = {(2, 2):-0.5+0.5*w, (6,6):-0.5+0.5*w, (3,3):-1., (7,7):-1., 
     (2,7):2., (2,6):-w, (3,6):2., (3,7):2.}
print('Q:{}'.format(Q))
result = sampler.sample_qubo(Q, num_reads=1000)

# look at the statistics of the samples
print('a a b c')

R = iter(result)
E = iter(result.data())
for line in result:
    sample = next(R)
    data = next(E)
    energy = data.energy
    occurrences = data.num_occurrences
    print(sample[2], sample[6], sample[3], sample[7], occurrences)

# Exercise: Can you test the sample statistics with different chain strength?


# Embedding tools
# Ocean's embedding package `minorminer`
# In this example, we are going to show how we can embed a graph using Ocean's `minorminer` package.
# As a reminder, a complete graph is a simple undirected graph in which every pair of distinct vertices is connected by a unique edge.

# import package from dwave-system
from minorminer import find_embedding

# define a K_5 graph
S_size = 5
S = {}
for i in range(S_size):
    for j in range(S_size):
        S[(i, j)] = 1

# Get the set of couplers from our sampler
A = sampler.edgelist

# Embed our problem onto our set of couplers
embeddings = find_embedding(S, A, verbose=2, random_seed=100)

# inspect the embeddings
print(embeddings)
print(len(embeddings))


from IPython.display import IFrame
IFrame("images/k5.pdf", width=600, height=300)


# Ocean's embedding tool `EmbeddingComposite`
# Using the `EmbeddingComposite` tool, we can combine solving and embedding in one step.

# For example, we can embed and solve our QUBO that we established in the beginning of this notebook for the Exactly-One-Truth Problem.
# As a reminder, this QUBO or energy function is given by : $$E(a,b,c) = 2ab + 2ac + 2bc - a - b -c.$$

# Import the EmbeddingComposite tool
from dwave.system.composites import EmbeddingComposite

# Define our QUBO dictionary
Q = {('a','b'): 2, ('a','c'): 2, ('b','c'): 2, ('a','a'):-1, ('b','b'):-1, ('c','c'): -1}

# Set up our sampler
sampler = EmbeddingComposite(DWaveSampler())   

# Run our QUBO through the EmbeddingComposite tool to the QPU and return our results
result = sampler.sample_qubo(Q, chain_strength=3, num_reads=1000)

# Print our results
print('a b c')
R = iter(result)
E = iter(result.data())
for line in result:
    sample = next(R)
    data = next(E)
    energy = data.energy
    occurrences = data.num_occurrences
    print(sample['a'], sample['b'], sample['c'], occurrences)


# Examples of Embeddings: 

# Largest bipartite graph with chain_length = 4

from IPython.display import IFrame
IFrame("images/largest_bipartite_chain4.pdf", width=600, height=300)


# Largest bipartite graph with any chain length

from IPython.display import IFrame
IFrame("images/largest_bipartite.pdf", width=600, height=300)


# Largest complete graph with chain_length=4

from IPython.display import IFrame
IFrame("images/largest_clique_chain4.pdf", width=600, height=300)


# Largest complete graph with any chain length

from IPython.display import IFrame
IFrame("images/largest_clique.pdf", width=600, height=300)

# shutdown the kernel
get_ipython().run_line_magic('reset', '-f')

