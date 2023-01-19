#=
Training Routine
=#

import gnn_test
import gnn_test: GNNDense
import Flux
import Zygote
import NNlib
import GraphNeuralNetworks as GNN
import GraphNeuralNetworks: GNNGraph
import Random

n_nodes = 10

Adj = Random.rand(0:1,n_nodes,n_nodes)
for ii=1:n_nodes
    A[ii,ii] = 0
end

n_edges = sum(Adj)

g = GNN.GNNGraph(
    Adj;
    ndata=(x=rand(1,n_nodes),),
    edata=(e=rand(1,n_edges),)
)

D = GNNDense(1 => 10, 1=> 15)

D(g)

model = GNN.GNNChain(
    GNNDense(1=>10, 1=>15),
    GNNDense(10=>10, 15=>15),
    GNNDense(10=>2, 15=>2, identity; bias=false)
)

ps = Flux.params(model)
opt = Flux.Adam(1f-4)

loss(g::GNN.GNNGraph) = abs(sum(model(g).ndata.x) - 5)

# Training loop
for epoch in 1:1000
    graddir = Zygote.gradient(()->loss(g), ps)
    Flux.Optimise.update!(opt,ps,graddir)

    if epoch % 10 == 0
        println("Loss at epoch $epoch: $(loss(g))")
    end
end