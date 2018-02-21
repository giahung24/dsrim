require 'tools/toolkit'
require 'torch'

local simulator = torch.class("Simulator")

function Simulator:__init(config)
end

function Simulator:get_relevance_Q_map(nb_Q, nb_D) -- a map of rel doc:  Q[index] -> { D[index] }
    -- for now, just simulation

    local relevance_for_each_Q = {}
    for i=1,nb_Q do
        local clicked_D_of_this_Q = {} -- list of doc index
        for k=1,5 do  
            local doc = math.random(nb_D) -- get a random D as clicked doc
            table.insert(clicked_D_of_this_Q, doc)
        end
        table.insert(relevance_for_each_Q, clicked_D_of_this_Q)
    end
    return relevance_for_each_Q;
end

function Simulator:negative_sample_index(whiteList, maxIndex, nNeg) -- return list of nNeg index, from 1 to maxIndex
    math.randomseed(os.time())
    local nNeg = nNeg or 4
    local list_out = {}
    while table.getn(list_out) < nNeg do
        local neg_doc = math.random(maxIndex)
        while has_value(whiteList,neg_doc) or has_value(list_out,neg_doc) do
            neg_doc = math.random(maxIndex)
        end
        table.insert(list_out, neg_doc)
    end
    return list_out
end