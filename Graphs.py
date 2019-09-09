
# @bhaney44

# Virtual Graph Composite

# The virtual graph composite is included within the dwave-system package within the Ocean SDK.
# Virtual graphs allow us to use an extended range of values for our coupler strengths in order to minimize the number of chains breaking for a given embedding.
# When you submit an embedding and specify a chain strength using these tools:
    # they automatically calibrate the qubits in a chain to compensate for the effects of biases that may be introduced as a result of strong couplings.

# Defining our problem

# First, we define our problem of interest.
# We will define a random Ising problem with a complete logical graph.


# define a complete graph
import numpy as np
S_size = 24
h = {v: 0. for v in range(S_size)}
J = {(a,b): np.random.choice([-0.5, 0.5]) for a in range(len(h)) for b in range(a+1, len(h))}
print("Assigned", len(h), "logical nodes.")
print("Assigned", len(J), "logical edges.")


# Setting up our sampler

# First, we define our sampler - we will use the QPU for this notebook.
# Let's look at the range of values available with the current QPU chip available.


# import package for communicating with the QPU:  dwave-system
from dwave.system.samplers import DWaveSampler
dwave_sampler = DWaveSampler(solver={'qpu': True})
print("Coupler strength range:", dwave_sampler.properties['extended_j_range'])


# Finding an embedding
# To use the virtual graph composite tool, we first need to generate an embedding for the graph of interest.
# We call `find_embedding` to create the embedding onto the working graph of the QPU.

# import package for graph minor embedding: minorminer
from minorminer import find_embedding

# Get the set of couplers from our sampler
A = dwave_sampler.edgelist

# Embed our problem onto our set of couplers
embedding = find_embedding(J, A)
chain_lengths = [len(embedding[node]) for node in range(S_size)]
long_chain = max(embedding, key=lambda k: len(embedding[k]))
print("Max chain length", len(embedding[long_chain]), "at node", long_chain, "\n\nEmbedding:\n")
for key,val in embedding.items(): print(key,":\t",val)


# Running on the QPU

# To see exactly how often each chain is breaking in a standard run on the QPU, we will first run our Ising problem on the QPU using the basic `embed_ising` method.
# We will use the optional parameter `chain_strength` within `embed_ising`, and will set this value to $2.0$ for our comparison.

# To use this tool, we provide the graph minor embedding discovered above to create and embedded version.
    # Then send this embedded ising problem over to the QPU using `sample_ising`.

from dwave.embedding import embed_ising
embedded_h, embedded_J = embed_ising(h, J, embedding, dwave_sampler.adjacency, chain_strength=2.0)
reads=1000
fixed_response = dwave_sampler.sample_ising(embedded_h, embedded_J, num_reads=reads)
energies = fixed_response.record.energy
print("QPU call complete using", fixed_response.info['timing']['qpu_access_time']/1000000.0, "seconds of QPU time.\n")


# Next we will look at the frequency that chains broke in our samples.
# This will be our metric to explore the advantages of `VirtualGraphComposite`.

from dwave.embedding import chain_break_frequency
chain_break_frequency = chain_break_frequency(fixed_response, embedding)
for key,val in chain_break_frequency.items():
    print("Chain", key, "broke in", val*100, "percent of samples.")


# Let's visualize these results by plotting each chain against the percentage of samples in which it broke.

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

x_vals = range(S_size)
x_pos = np.arange(len(chain_break_frequency.items()))
y_vals = [-1 for _ in x_vals]
for key, val in chain_break_frequency.items(): y_vals[key] = val*100
    
f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)
ax.bar(x_pos, y_vals)
ax.set_xticks(x_pos, x_vals)
ax.set_ylabel('Percentage of samples')
ax.set_xlabel('Node Chain')
ax.set_title('Percentage of Samples with Broken Chain')

bx = f.add_subplot(122)
bx.scatter(chain_lengths, y_vals)
bx.set_xlabel('Chain Length')
bx.set_title('Break Frequency vs Chain Length')


# Running with `FixedEmbeddingComposite`

# Now we will run the same problem using `FixedEmbeddingComposite`.
# This tool will use the embedding we found earlier to send our problem over to the sampler with `chain_strength=1.0`.

from dwave.system.composites import FixedEmbeddingComposite
fe_response = FixedEmbeddingComposite(dwave_sampler, embedding).sample_ising(h, J, num_reads=reads)
energies = fe_response.record.energy
print("QPU call complete using", fe_response.info['timing']['qpu_access_time']/1000000.0, "seconds of QPU time.")
print("Energy values in range", [energies.min(), energies.max()])


# Next we will visualize our results.
# `FixedEmbeddingComposite` and `VirtualGraphComposite` return information about broken chains by providing the `chain_break_fraction` in the response object.  Each sample has an associated chain break fraction that indicates the percentage of chains that broke in that sample.

fe_x_vals = range(reads)
fe_x_pos = np.arange(reads)
fe_y_vals = []
for datum in fe_response.data(): 
    for _ in range(datum[2]): fe_y_vals.append(datum[3])
    
plt.plot()
plt.scatter(fe_x_pos, fe_y_vals)
plt.ylabel('Chain Break Fraction')
plt.xlabel('Sample')
plt.ylim([0.0,1.0])
plt.title('Chain Break Fractions')
fe_average = np.array(fe_y_vals).mean()
plt.plot([0, reads], [fe_average,fe_average], color='red')
print("Average chain break fraction", fe_average)


# Running with `VirtualGraphComposite`

# Next we will run our same problem using the `VirtualGraphComposite` tool.
# This tool may take longer to run than `FixedEmbeddingComposite` as it may need to calculate flux bias offsets (discussed further below).

# When we use `VirtualGraphComposite` we specify the parameter `chain_strength`.
# The maximum value for this parameter is `chain_strength=2.0`.

# The range of coupling strengths available is finite.
# Chaining is typically accomplished by setting the coupling strength to the largest allowed negative value and scaling down the input couplings relative to that.
# Yet a reduced energy scale for the input couplings may make it harder for the QPU to find global optima for a problem.


from dwave.system.composites import VirtualGraphComposite
vg_response = VirtualGraphComposite(dwave_sampler, embedding, chain_strength=2.0).sample_ising(h, J, num_reads=reads)
energies = vg_response.record.energy
print("QPU call complete using", vg_response.info['timing']['qpu_access_time']/1000000.0, "seconds of QPU time.")
print("Energy values in range", [energies.min(), energies.max()])


# Visualize these results.

vg_x_vals = range(reads)
vg_x_pos = np.arange(reads)
vg_y_vals = []
for datum in vg_response.data(): 
    for _ in range(datum[2]): vg_y_vals.append(datum[3])
    
plt.plot()
plt.scatter(vg_x_pos, vg_y_vals)
plt.ylabel('Chain Break Fraction')
plt.xlabel('Sample')
plt.ylim([0.0,1.0])
plt.title('Chain Break Fractions')
vg_average = np.array(vg_y_vals).mean()
plt.plot([0, reads], [vg_average,vg_average], color='red')
print("Average chain break fraction", vg_average)

# Now we can examine these two plots side by side to see how they compare.

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(vg_x_pos, vg_y_vals, c='b', marker="s", label='VG')
ax1.scatter(fe_x_pos, fe_y_vals, c='r', marker="o", label='FE')
plt.legend(loc='upper left')
print("Fixed Embedding Chain Break Fraction Average: \t", fe_average)
print("Virtual Graph Chain Break Fraction Average: \t", vg_average)

# Scaling
 
# If instead our Ising problem has values that lie outside of this range, we need to scale these values down.

h = {v: 0. for v in range(S_size)}
J = {(a,b): np.random.choice(range(10)) for a in range(len(h)) for b in range(a+1, len(h))}
print("Assigned", len(h), "logical nodes.")
print("Assigned", len(J), "logical edges.")


# Find the minimum and maximum values in our set of $h$ and $J$ values.
min_h_val = min(zip(h.values()))[0]
max_h_val = max(zip(h.values()))[0]
print("h value range:", [min_h_val, max_h_val])

min_J_val = float(min(zip(J.values()))[0])
max_J_val = float(max(zip(J.values()))[0])
print("J value range:", [min_J_val, max_J_val])


# Figure out what our scaling factor is
J_scaling_factor = max(min_J_val/-1.0, max_J_val/1.0)
h_scaling_factor = max(min_h_val/-1.0, max_h_val/1.0)
scaling_factor = max(J_scaling_factor, h_scaling_factor)
print("For scaling, we can to divide all values by", scaling_factor)


# Scale values down to correct size range
scaled_J = J.copy()
scaled_h = h.copy()
for key in h: scaled_h[key] = scaled_h[key]/scaling_factor
for key in J: scaled_J[key] = scaled_J[key]/scaling_factor
print("h values in range", [min(zip(scaled_h.values()))[0],max(zip(scaled_h.values()))[0]])
print("J values in range", [min(zip(scaled_J.values()))[0],max(zip(scaled_J.values()))[0]])


# Now we can send our new problem over to the QPU using `VirtualGraphComposite`.


scaled_vg_response = VirtualGraphComposite(dwave_sampler, embedding, chain_strength=2.0).sample_ising(scaled_h, scaled_J, num_reads=reads)

energies = scaled_vg_response.record.energy
print("QPU call complete using", scaled_vg_response.info['timing']['qpu_access_time']/1000000.0, "seconds of QPU time.")
print("Energy values in range", [energies.min(), energies.max()])


# Note for using `VirtualGraphComposite`


print("Coupler strength range:", dwave_sampler.properties['per_qubit_coupling_range'])


# When you run the block below demonstrating this problem, you should see an error statement: `SolverFailureError: Total coupling -12.000 on qubit 1372 is out of range (-9.0 6.0).`

h_fail = {129: 0.0}
J_fail = {(1,129): -2.0, (129,132):-2.0, (129,133):-2.0, (129,134):-2.0, (129,135):-2.0, (129,257):-2.0}
embedding_fail = find_embedding(J_fail, dwave_sampler.edgelist)
if not embedding_fail:
    print("Minor embedding hueristic failed this time. Please rerun this cell.")
else:
    fail_vg_response = VirtualGraphComposite(dwave_sampler, embedding_fail, chain_strength=2.0).sample_ising(h_fail, J_fail, num_reads=reads)


# Flux Bias Offsets

# In an optimal QPU calibration, annealing an unbiased chain produces spin-state statistics that are equally split between spin-up and spin-down.

# The optimal offset value for a chain depends on the qubits and couplers involved and on the chain coupling strength.

# The default behavior of `VirtualGraphComposite` is that flux biases are pulled from the database or calculated empirically.
# If you wish to manually set your own flux bias offsets, there is an optional parameter that you can use. 

# For more information on flux bias offsets, take a look at our documentation [here](https://docs.dwavesys.com/docs/latest/c_qpu_0.html#fb-sigmoid).
