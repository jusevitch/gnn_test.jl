#=
Extras to augment the GraphNeuralNetworks library.
=#


struct GNNDense{D1,D2} <: GNN.GNNLayer
    node_dense::D1
    edge_dense::D2
end

Flux.@functor GNNDense

function GNNDense(
    node_dims::T1,
    edge_dims::T2,
    activation=NNlib.leakyrelu;
    bias=true,
    init=Flux.glorot_uniform
) where {T1 <: Pair, T2 <: Pair}

    return GNNDense(
        Flux.Dense(node_dims, activation; bias=bias, init=init),
        Flux.Dense(edge_dims, activation; bias=bias, init=init)
    )
end

function (D::GNNDense)(g::GNN.GNNGraph)
    ndata = D.node_dense(g.ndata.x)
    edata = D.edge_dense(g.edata.e)

    return GNN.GNNGraph(g, ndata=ndata, edata=edata)
end