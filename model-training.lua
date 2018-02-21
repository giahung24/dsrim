require 'nn'
nninit = require 'nninit'
require 'optim'
-- local package
-- package.path = package.path .. ";../?.lua"
require 'tools/toolkit'
require 'tools/simulator'
require 'tools/preprocessing'
require "InputStream"
require 'MarginQuadRankingCriterion'

--=========================================================
local function create_model(input_size,hidden_layers_size,last_layer_vector_size, drop_out)
    local Q_layer = nn.Sequential()
    Q_layer:add(nn.Linear(input_size, hidden_layers_size):init('weight', nninit.xavier,{dist='uniform',gain=math.sqrt(3)}))
    Q_layer:add(nn.ReLU())
    Q_layer:add(nn.Linear(hidden_layers_size, hidden_layers_size):init('weight', nninit.xavier,{dist='uniform',gain=math.sqrt(3)}))
    Q_layer:add(nn.ReLU())
    Q_layer:add(nn.Dropout(drop_out))
    Q_layer:add(nn.Linear(hidden_layers_size, last_layer_vector_size):init('weight', nninit.xavier,{dist='uniform',gain=math.sqrt(3)}))
    Q_layer:add(nn.ReLU())

    local D_layer =  Q_layer:clone('weight', 'gradWeight', 'bias', 'gradBias')

    local parallelBranch = nn.ParallelTable()
    parallelBranch:add(Q_layer)
    parallelBranch:add(D_layer)

    local model = nn.Sequential()
    model:add(parallelBranch)
    model:add(nn.CosineDistance())

    return model
end

---========================================================
cmd = torch.CmdLine()
cmd:option('-k',0,'SCENARIO_K_FOLD from 0 to 9')
cmd:option('-min_max','centroid','min, centroid or max')  --  the choice of the representative object R(cj) [ref.Section 4.2 in the paper]
cmd:option('-k_cluster',200,'Num of cluster 50, 100, 200')  -- [ref.Section 4.2 in the paper]
cmd:option('-dropout',0.3,'DROPOUT (0.3 by default)')
cmd:option('-l2norm', 0, 'L2 LAMBDA (2e-5 for example, 0 by default)')
cmd:option('-learning_rate', 1e-5, 'LEARNING RATE OF GRAD DESC (1e-5 by default)')
cmd:option('-hidden_size', 64, 'HIDDEN LAYERS SIZE (256 by default)')
cmd:option('-output_size', 32, 'LAST LAYER SIZE (128 by default)')
cmd:option('-loadmodel', false, 'true or false (by defaut)')
params = cmd:parse(arg or {})

--------------- model hyper-param --------------------
local SCENARIO_K = params['k']
local min_max = params['min_max']
local k_cluster = params['k_cluster']

local learning_rate = params['learning_rate']
local l2_lambda = params['l2norm']
local hidden_layers_size = params['hidden_size']
local last_layer_vector_size = params['output_size']
local drop_out = params['dropout']
local loadModel = params['loadmodel'] -- if true, model will be loaded from 'model_path' below


local MODEL_NAME = "semText-k"..SCENARIO_K.."."..hidden_layers_size.."_"..last_layer_vector_size..".drop"..drop_out..".gov"

---------------- LOAD DATASET --------------------------
--[[
    View InputStream.lua for more info.
--]]
local preparedPath = "........................................."
local istream = InputStream(SCENARIO_K, preparedPath, 'gov', min_max, k_cluster)

local input_size = 100 + k_cluster

--------------------------------------------------------
local model = nil
local model_path = 'model.'..MODEL_NAME..'.saved'

if loadModel then
    model = torch.load(model_path)
else
    model = create_model(input_size,hidden_layers_size,last_layer_vector_size,drop_out)
end


local parameters,gradParameters = model:getParameters()

-- local label = torch.Tensor(5):fill(0)
-- label[1]=1 --label: [1,0,0,..,0,0]
-- criterion = nn.CrossEntropyCriterion() --passing label into param => get loss of D+ only

local label = 1 --torch.Tensor(5):fill(1)  -- label for MarginRankingCriterion
criterion = nn.MarginQuadRankingCriterion(1)

print( 'model name:' .. MODEL_NAME )
--------------------OPTIM METHOD--------------------------------------
local optimName = 'adam' ---- sgd || adam || adadelta

local optimState = {}
local optimMethod = nil
if optimName == 'sgd' then
    optimState = {
        learningRate = learning_rate,
        momentum = 0.8,
        learningRateDecay = 1e-6
    }
    optimMethod = optim.sgd
elseif optimName == 'adam' then
    optimState = {
        learningRate = learning_rate,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8
    }
    optimMethod = optim.adam
elseif optimName == 'adadelta' then
    optimState = {
      rho = 0.6,
      eps = 1e-8,
    }
    optimMethod = optim.adadelta
else error('unknown optimization method') 
end


---------------------- TRAINING --------------------------
local epoch = 0
local maxEpoch = 100
local total_loss = 0

local model_output = nil
local loss_train = nil
local trainingPair = 0


-- print('Train with optim method:',optimName)

print(string.format("epoch\ttotal_train_loss\tepoch_train_loss\tepoch_test_loss\n"))

local min_test_loss = 9999
while(epoch < maxEpoch) do
	epoch = epoch + 1
    model:training()
    local epoch_train_loss = 0
    local epoch_train_pair = 0
    local skip_pair = 0
    while istream:hasNextTrainPair() do
        local pool = istream:nextTrainPair() --{Q,D+,D-,D-,D-,D-}
        if pool==nil then
            skip_pair = skip_pair + 1
        else
			epoch_train_pair = epoch_train_pair + 1
			trainingPair = trainingPair + 1
        	local q_tensor = pool['Q']:repeatTensor(5,1)
        	local d_tensor = torch.Tensor(5,input_size)
            -- print(input_size)
            -- print(#pool['D+'])
       		d_tensor[1] = pool['D+']
       		d_tensor[2] = pool['D1-']
       		d_tensor[3] = pool['D2-']
       		d_tensor[4] = pool['D3-']
       		d_tensor[5] = pool['D4-']
            local input = {q_tensor,d_tensor}

       		------ netword forwarding
            local feval = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end
                gradParameters:zero()
                ---- get the cosine similarity for all positive and negtive.
                
                local output = model:forward(input)
                for i = 1, #output:storage() do

                	if (output[i]) ~= output[i] then
                		print("nan at d"..i)
                		-- print(d_tensor[i])
                		print( istream.trainPairs[istream.__indexTrain-1] )
                	end
                end
                -- print(string.format( "Output:" ))
                -- print( output )
                -- print( output )
                ---- forward the criterion
                local err = criterion:forward(output,label)
                local do_df = criterion:backward(output,label)
                model:backward(input, do_df)
                -- penalty L2
                local norm = torch.norm
                err = err + l2_lambda * norm(parameters,2)^2/2
                gradParameters:add( parameters:clone():mul(l2_lambda) )
                
                total_loss = total_loss + err
                epoch_train_loss = epoch_train_loss + err
                -- print(string.format("Loss: %-s", err))
                return err, gradParameters
            end ---- end feval

            optimMethod(feval, parameters, optimState)
        end
    end -- end a training epoch
    local total_pair = skip_pair + epoch_train_pair
    io.stderr:write(string.format( "Skip %-s pairs in total %-s pairs\n", skip_pair, total_pair ))
    -----------------------------------------------------------
    ----- CHECK DEVSET LOSS
    local test_loss = 0
    local testPair = 0
    model:evaluate()
    while istream:hasNextDevPair() do
        local pool = istream:nextDevPair() --{Q,D+,D-,D-,D-,D-}
        if pool ~= nil then

            -- local input = {{pool['Q'],pool['D+']},{pool['Q'],pool['D1-']},{pool['Q'],pool['D2-']},{pool['Q'],pool['D3-']},{pool['Q'],pool['D4-']}}
            local q_tensor = pool['Q']:repeatTensor(5,1)
            local d_tensor = torch.Tensor(5,input_size)
            d_tensor[1] = pool['D+']
            d_tensor[2] = pool['D1-']
            d_tensor[3] = pool['D2-']
            d_tensor[4] = pool['D3-']
            d_tensor[5] = pool['D4-']
            local input = {q_tensor,d_tensor}
            local output = model:forward(input)
            local test_err = criterion:forward(output,label)
            -- local norm = torch.norm
            -- test_err = test_err + l2_lambda * norm(parameters,2)^2/2
            test_loss = test_loss + test_err
            testPair = testPair + 1
        end
    end -- end a test monitoring epoch
    istream:resetDev()

    local epoch_test_loss = test_loss/testPair
    print(string.format("epoch %-s\t%-s\t%-s\t%-s", epoch, total_loss/trainingPair, epoch_train_loss/epoch_train_pair, epoch_test_loss))

    if epoch_test_loss < min_test_loss then
        min_test_loss = epoch_test_loss
        -- print(string.format("Saving model at epoch %-s...", epoch))
        model:clearState()
        torch.save(model_path, model)
    end
    istream:resetTrain() -- reset streaming and shuffle 
end -- end an epoch



--[[
    In prediction phase, to compute cosine similarity of (Q,D)
local output = model:forward({queryVector, docVector})
local neural_score = output[1]
--]]