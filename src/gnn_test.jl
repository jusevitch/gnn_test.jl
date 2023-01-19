#=
Simple module for testing variable sized GNNs
=#

#=
Notes:

Transformers: 
* Graph is fully connected
* Edge weights determined by attention mechanism
=#

module gnn_test


import Flux
import GraphNeuralNetworks as GNN
import NNlib
import Zygote

# includes
include("./gnn_extras.jl")

end
