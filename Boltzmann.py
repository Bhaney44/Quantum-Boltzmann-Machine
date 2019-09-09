
#BHANEY44_@B$
#@B$_BOLTZMANN

# Introduction to Boltzmann machines

# Boltzmann machines are an expressive family of models allowing the representation of a probability distribution over a set of binary variables. 
# These are often used to learn a stochastic, generative model over inputs.
# An undirected graph is used to represent conditional dependencies between variables.

# It is convenient to partition the variables into two sets of variable 
    #Visible, v
    #Hidden, h

# Visibles usually correspond to variables of interest to the applications.
    # For example, the inputs and outputs of our addition problem.
    
# Hiddens are additional variables enabling the encoding of more complex dependencies amongst visible variables.

# By restricting the graphical structure allowed for the Boltzmann machine, we can formulate tractable sampling and training algorithms.

# Popular BM architectures include:
 
    #Restricted Boltzmann machines (RBMs), 
        #no lateral edges within the visibles or hiddens, bipartite graphs.

    #Semi-restricted Boltzmann machines
        #which allows lateral edges within the hiddens, but not the visibles.

    #Deep Boltzmann machines (DBMs)
        #which consist of multiple RBMs tied together such that the visibles of one form the hiddens of the next.


# The probability distribution encoded by a BM can be represented by an energy function defined over the visible and hidden variables. 


# Model design

# This notebook introduces the reference example for training an RBM on the D-Wave machine.

# Defining the problem

# Integer addition is a ubiquitous operation.

    # The addition operation can be represented by a table of data, where each row corresponds to one example: an allowed combination of inputs and outputs. 
    # The allowed combinations of variables can be characterized by a distribution over the variables, with narrow peaks around the allowed values.

    # If we can learn the distribution, we can use it to solve the addition problem. 
    # To perform addition, we simply fix the variables corresponding to the input operands and sample from the conditional distribution over the outputs.
 
    # Normally, addition is not thought of as a learning problem. Well-understood deterministic algorithms exist and can be optimized at the hardware level. 
    # But learning a distribution over the data allows us to do additional things. 
    # For example, by clamping the output and one of the inputs, we can sample from the conditional distribution over the other input, yielding the answer to a subtraction problem. 
    # Similarly, we can clamp the output only and obtain a multi-modal distribution peaking at all those inputs that yield the fixed output.

#Setting up our addition model

# We consider a 3-bit adder.
    #It means that there are 10 visibles: two inputs that each have three bits and one output of four bits.

# Placing the units on Chimera
    # From the problem definition, a 3-bit adder requires one (half adder) and two (full adders). 
    # To place them on a Chimera graph, we refer to the following figure:


from dwave_bm.rbm import BM

# Model dimensions.
num_visible = 10

# the hidden layer is a bipartite graph of 9 by 5
num_hidden = [9, 5] 


# Create the Boltzmann machine object
r = BM(num_visible=num_visible, num_hidden=num_hidden)
r


# Exploring the model

# We have created a Boltzmann machine. 




# Here v and h correspond to the bit vectors of visibles and hiddens, respectively. 
# These are the variables over which we want to learn the distribution.


# Upon construction these parameters were populated with random values.
# If we already know a set of good parameters, we could provide these to the `r` object.

# We explore the shapes of the parameter vectors below.
# If all has gone as planned, they should easily follow from the model dimensions specified on `BM` creation.

# Dimensions

# Bias vector to visibles
print("b: {}".format(r.b_vect.shape))

# Bias vector to hiddens
print("c: {}".format(r.c_vect.shape))

# Weights between visibles and hiddens
print("W: {}".format(r.weights.shape))

# Lateral connections in hidden layer
print("X: {}".format(r.X.shape))


# Sampling from the Boltzmann machine

# Now that the BM is created and has some parameters (even if they are random), we can sample from it.

# For now we simply sample from all the visibles, which means sampling the binary random variables from a distribution


# where n is the number of visibles.
# Note that the BM represents a model over all variables, including the hiddens, we can sample the hidden variables also.
# This is usually not required for the application in mind.

# To sample from all the visibles, we use the [`BM.sample_clamped_model`](documentation/dwave_bm.rbm.bm_class.html#dwave_bm.rbm.bm_class.BM.sample_clamped_model) function.

# Do not clamp any bits
clamped_bits = []

# Draw the samples
samples = r.sample_clamped_model(clamped_bits)

print "Obtained {} samples. The first 5 samples are:".format(len(samples))
print samples[0:5,:]


# The call to [`BM.sample_clamped_model`](documentation/dwave_bm.rbm.bm_class.html#dwave_bm.rbm.bm_class.BM.sample_clamped_model)
#should have returned a number of samples distributed over the encoded probability distribution.
# Currently we do not expect much structure in the distribution, and we can calculate some statistics to confirm this.

# For example, we may estimate the mean and standard deviation of each variable.


import numpy as np

N = len(samples)

# Calculate the mean of each variable by summing rows
means = sum([samples[i,:] for i in range(N)]) / float(N)

# Calculate the standard deviation of each variable
# with the row of means calculated above

stddevs = np.sqrt(sum([(samples[i,:] - means)**2 for i in range(N)]) / float(N))

print("means = {}".format(means))
print("standard deviations = {}".format(stddevs))


# For our untrained BM, we would expect no combination of variable assignments to be favoured over any other.
#In that case, and since these are binary variables, we expect a mean near 0.5 and a standard deviation near 0.5.

# If the samples we received from the untrained BM `r` conform to this, all should be in order.
#Next, we will train the BM.

# Training the hand-crafted semi-RBM with the QPU

# In the previous example we designed the structure of our addition model and introduced the ability to sample from it. The samples we obtained from the untrained BM were, predictably, not of much use.

# In this notebook we will train the BM on some data.

# Setting up our semi-RBM

# We consider a 3-bit adder. This means that there are 10 visibles, two inputs $a$ and $b$ that each have three bits and one output $s$ of four bits.

# We can then import the `BM` class from the `rbm` module.
# This class encodes the graphical representation of the BM and houses the methods for training it.
# We set up an instance `r` of the structure we require.

# In this example, we use our domain-specific knowledge of the problem (addition), to design a BM with a better initial guess.

# First, we need to load a configuration file for the D-Wave quantum computer.

# JSON config file with hardware credentials
# and structural information
import json
import os
hw_config_file_sample = 'rbm_configs.json'

# Set location for the configuration file to current working directory (CWD), if writeable, or /tmp/
write_base_path = os.getcwd()
if not os.access(write_base_path, os.W_OK):
    write_base_path='/tmp/'
hw_config_file = os.path.join(write_base_path, 'rbm_configs_example.json')

# define url, token and solver_name for running jobs on the hardware
url = 'https://cloud.dwavesys.com/sapi'
token = ''
solver_name = ''

with open(hw_config_file_sample, 'r') as cf:
    hw_config = json.load(cf)
    
# overwrite url, token and solver_name in the config 
hw_config['url'] = url
hw_config['token'] = token
hw_config['solver'] = solver_name

# save to a new json file 
with open(hw_config_file, 'w') as cf:
    json.dump(hw_config, cf, indent=4) 

# print the config file
with open(hw_config_file, 'r') as cf:
    for line in cf.readlines():
        print line,


# Loading the hand-crafted model and place it on the hardware

# Here we load the model we designed earlier.
# The designed model is available in the DIMACS format, a common format for specifying graph problems. Then we place it on the hardware.

# The file storing the designed model
model_file = 'dwave_bm/dimacs/qubo-sample.txt'

# Since the model was designed to fit into the native QPU graph,
# we use the BMPlacer class to load the problem graph.

from dwave_bm.rbm import BM, BMPlacer
placer = BMPlacer(num_visible, num_hidden, json_file=hw_config_file)
model = placer.read_model_dimacs_qubo(model_file)


# Now we are ready to create a BM object and initialize it to our hand-crafted model.



# Create our BM object and pass it the designed model
r = BM(num_visible=num_visible, num_hidden=num_hidden,
       W_mask=placer.W_mask, X_mask=placer.X_mask,
       model=model)
r


# Loading the training data

# We use `numpy` to represent and manipulate tabular data.

# The examples for the 3-3-4 addition operation have been collected previously and are loaded below. Note how we find a random permutation of the examples before proceeding.

m = np.load('dwave_bm/data/3-3-4.npz')
data_pos = m['data_pos']

data_pos = data_pos[np.random.permutation(data_pos.shape[0])]


# The first five lines of the data look as follows.

data_pos[:5]


# Converting to decimal base, the first five examples are shown below. The final 4-bit entry in each example is the sum of the first two 3-bit entries.


from dwave_bm.example_utils import b2d

[map(b2d, [ex[0:3], ex[3:6], ex[6:10]]) for ex in data_pos[:5]]


# Sampling

# To set up our BM to use the D-Wave quantum computer as a sampler, we may execute the code below.


# Valid samplers are 'cd-10', 'pcd-10', 'hw'
sampler = 'hw'


# Training the Boltzmann machine

# Now that we have a family of models and some examples, we can train our model on the data. Before we start training, we add a logger that captures some information about the training process and intermediate results.


# Function subscribe_logger writes the state of the BM out to
# disk (perhaps when )
from dwave_bm.example_utils import subscribe_logger

save_model = True
save_fig = False

# define a folder to save your result
result_folder = '/tmp/results/test_' + sampler

subscribe_logger(r, result_folder)

# This is how often we want to run the logger.
log_frequency = 10


# Now we set up some hyperparameters for our training session.

# The maximum number of epochs to train.
    # epochs = 10001
epochs = 101

# Minibatch size and number of samples are used to estimate the
# gradient of the cost function.

minibatch_size = 64

# Learning rate and momentum are used to calculate the next point
# \theta after the gradient is known.

eta0 = 0.5
learning_rate = [eta0*(1 / (1 + i / 200.0)) for i in range(epochs)]
momentum = 0.5


# We now have all the information we need to start the training process.
# Executing the next cell will take some time.
# The logger we hooked into the training procedure above will write out the trained model to the disk at the frequency selected
    #(every 100 epochs is a good frequency).

# The method `BM.train` returns a list containing the RMSE after each logging step.

# Training BM on the D-Wave quantum computer may take some time.
    #Plan the following runs carefully!


# Train the BM. Takes some time.
# Uncomment the following code to train your BM on the D-Wave quantum computer when it is necessary
# For debug purpose, try using `cd` or `pcd` as the sampler

# r.train(data_pos,
#         minibatch_size=minibatch_size,
#         max_epochs=epochs,
#         learning_rate=learning_rate,
#         momentum=momentum,
#         sampler=sampler,
#         part_l=placer.part_l,
#         part_r=placer.part_r,
#         save_model=save_model,
#         save_fig=save_fig,
#         hw_config_file=hw_config_file,
#         every_n_epochs=log_frequency,
#         )

#-----------------------------------

# Exploring the QPU-trained semi-RBM

# Previously, we set up and trained an semi-RBM on data describing the 3-bit addition problem.
# We formed the model and seeded its parameters with a QUBO problem built from constraints describing addition at the bit level. If this procedure was successful, we should now have a model of the distribution over the data.

# We employed the QPU to generate the samples needed to train the semi-RBM. Previous efforts at employing the PCD sampling mechanism failed for this model.

# Next, we will explore the trained RBM and the distribution it represents.



# Loading the trained model

# The logger we hooked into the training procedure should have written out the model at a frequency we picked. After setting up our session again and creating a new BM object, this time we load the last model saved during training. This should be the model that best fits the training data.


# Some boilerplate to set up the environment
get_ipython().run_line_magic('matplotlib', 'inline')

sampler = 'hw'

# The directory in which to look for results. 
# To use pre-generated results, set result_folder variable to 'results/hw'
result_folder = '/tmp/results/test_' + sampler

# A function to load the model saved during our previous training run
def load_model(epoch):
    from dwave_bm.example_utils import result_filename
    fn, result_epoch = result_filename(directory=result_folder,
                                       num_visible=num_visible,
                                       num_hidden=num_hidden,
                                       epoch=epoch,
                                       sampler=sampler
                                      )
    print "Loading model from '{}' for epoch '{}'".format(fn, result_epoch)
    return np.load(fn)

# The result epoch we would like to investigate
result_epoch = 'latest'

# We use the function defined above to load a model
m = load_model(result_epoch)

# The model is a numpy archive, which has several 'files' that
# parametrize the trained BM and some metadata
m.files


# First, we can plot how the RMSE of the learned model changed with passing epochs. Hopefully it improved sufficiently over the training epochs.

# import matplotlib for plotting
import matplotlib.pyplot as plt

domain = m['logged_epochs']
errors = m['errors']
plt.plot(domain, errors)


# The training error evolution looks better than it did for the PCD-trained model.
# There is a clear improvement over time (epoch).

# Now that our BM is trained, we can sample from the model it represents.
# For example, we can sample the sum bits while clamping the input bits to a value of interest.
# This allows us to sample from a particular instance of the addition problem.

# More specifically, we can sample from the distribution


# First, we define some convenient ways of loading a BM and sampling from its inputs.

# Create a BM from a model saved in a previous training session
def load_BM(epoch=None):
    from dwave_bm.rbm import BM
    
    # Load the saved model, using function defined above
    m = load_model(epoch)
    
    # Create a BM object seeded with the loaded model
    return BM(num_visible=num_visible, num_hidden=num_hidden, model=m)


# Sample from the passed BM, clamping the inputs
    # The BM to sample from
    # a = 5 by default
    # b = 3 by default
    
def sample_clamped_inputs(  r,              
                            a = [1, 0, 1],  
                            b = [0, 1, 1],  
                          ):
    
    # Gather the clamps. The inputs are arranged with a first, then b,
    # in the order of visibles.
    
    clamped_bits = a + b
    
    # We can now sample from the clamped model.
    samples = r.sample_clamped_model(clamped_bits)

    # Show off some of the samples we got
    print("Obtained {} samples. The first 5 samples are:".format(len(samples)))
    print(samples[0:5, :])
    
    # For binary to int conversion
    from dwave_bm.example_utils import b2d

    # Find the correct answer from the integer values represented
    # by the clamped visible inputs
    correct_s = b2d(a) + b2d(b)

    # Count the number of samples that corresponded to the correct answer
    hits = map(b2d, samples).count(correct_s)
    
    # Print out the hit rate
    print "A total of {}/{} samples gave the correct answer.".format(hits, len(samples))
        
    return samples



# Now that we have some convenient ways of loading and sampling
# BMs, we can start out by loading the BM we had at the end
# of our previous training session.

r_trained = load_BM('latest')

# Here we set the values to which to clamp the inputs
# a = 5
# b = 3

a = [1, 0, 1] 
b = [0, 1, 1]  

# We can then draw some samples from it.
samples_trained = sample_clamped_inputs(r_trained, a=a, b=b)


# If all has gone well, we should have achieved a reasonably high hit rate above.
# Unlike in the previous example, this indicates a narrow mode around the correct answer in the conditional distribution p(s|a,b)p(s|a,b).
# Again, we can gain a more complete picture of the conditional distribution by plotting histogram of the samples obtained.


# A function to plot frequencies of events
from dwave_bm.example_utils import plot_histogram, b2d

# Execute the function on the tranined samples
plot_histogram(map(b2d, samples_trained), range(15))


# There is, again, a narrow mode around the correct answer.
# Things are looking considerably better than with the PCD-trained model.

# We now investigate how the distribution evolved over training time.

# We can try some different clamps
a = [1,0,1]
b = [1,1,0]

# Do a snapshot of the probability distribution for a
# particular set of clamps at the specified epoch
def plot_at_epoch(epoch):
    r = load_BM(epoch)
    samples = sample_clamped_inputs(r, a=a, b=b)
    plot_histogram(map(b2d, samples), range(15), title="Epoch {}".format(epoch))

# Go over a few orders of magnitude
plot_at_epoch(0)
plot_at_epoch(100)

#By default, the epochs is set to 100 in section 3.5. If set to 10001, uncomment below to view graphs.
#plot_all_at_epoch(1000)
#plot_all_at_epoch(4000)


# With limited number of epochs in the previous training, the distribution also started clustered widely around the correct answer.
    # but it did not evolve to convincingly favour the correct answer.
# In this case the evolution looks more like we expected before, with the distribution gradually tightening around the correct answer.

# As before, we now plot the evolution of other conditional distributions.



# Calculate and plot the conditional distributions p(s|a,b), p(a|s,b) and p(a,b|s)
def plot_all_at_epoch(epoch):
    r = load_BM(epoch)
    r.plot_all()

plot_all_at_epoch(0)
plot_all_at_epoch(100)
#By default, the epochs is set to 100 in section 3.5. If set to 10001, uncomment below to view graphs.
#plot_all_at_epoch(1000)
#plot_all_at_epoch(4000)


# As in the single case examined above, the training seems to have been successful in this case.
# The correct answer was slightly favoured by the untrained (designed) model.
# training allowed the model to adjust its weights to tighten up the distribution, until it peaks sharply at each correct answer.

# We successfully trained an semi-RBM on the QPU!

# shutdown the kernel
get_ipython().run_line_magic('reset', '-f')

