require 'nn'
require 'optim'
local myUtil = require("../MyCommon/util.lua")

do
  local trainerPool = {}

  function trainerPool.getDefaultTrainParams(nRows, strOptimMethod, nMaxIteration)

    local taTrainParam = {  --batchSize = 9, 
                            batchSize = math.floor(nRows),
                            criterion = nn.MSECriterion(),
                            nMaxIteration = nMaxIteration or 10,
                            coefL1 = 0.0,
                            coefL2 = 0.0,
                            strOptimMethod = strOptimMethod or "CG",
                            isLog = true,
                            taOptimParams = {}
                          }

    if taTrainParam.strOptimMethod == "SGD" then
      taTrainParam.taOptimParams = { 
        learningRate = 0.9,
--        learningRateDecay = 0.999,
        momentum = 0.9 }
      taTrainParam.fuOptim = optim.sgd
  
    elseif taTrainParam.strOptimMethod == "LBFGS" then
      taTrainParam.taOptimParams = { 
        maxIter = 100,
        lineSearch = optim.lswolfe }
      taTrainParam.fuOptim = optim.lbfgs

    elseif taTrainParam.strOptimMethod == "CG" then
      taTrainParam.taOptimParams = {
        maxIter = 30 }
      taTrainParam.fuOptim = optim.cg

    else
      error("invalid operation")
    end

    return taTrainParam
  end

  function trainerPool.pri_trainSparseInputNet_SingleRound(mNet, taX, teY, taTrainParam)
    parameters, gradParameters = mNet:getParameters()
    local criterion = taTrainParam.criterion
    local overallErr = 0
		local scale = taTrainParam.taOptimParams.learningRate or 1

      local fuEval = function(x)
        collectgarbage()

        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- evaluate function for the complete mini batch
        local tePredY = mNet:forward(taX)
        local f = criterion:forward(tePredY, teY)

        -- estimate df/dW
        local df_do = criterion:backward(tePredY, teY, scale)
        mNet:backward(taX, df_do, scale)

       -- penalties (L1 and L2):
        if taTrainParam.coefL1 ~= 0 or taTrainParam.coefL2 ~= 0 then
          -- locals:
           local norm,sign= torch.norm,torch.sign
 
          -- Loss:
          f = f + taTrainParam.coefL1 * norm(parameters,1)
          f = f + taTrainParam.coefL2 * norm(parameters,2)^2/2

          -- Gradients:
          gradParameters:add( sign(parameters):mul(taTrainParam.coefL1) + parameters:clone():mul(taTrainParam.coefL2) )
        end
        
--        overallErr = overallErr + f

        return f, gradParameters
      end --fuEval

      taTrainParam.fuOptim(fuEval, parameters, taTrainParam.taOptimParams)

    return trainerPool.getErr(mNet, taX, teY, taTrainParam)
  end

  function trainerPool.getErr(mNet, taInput, teTarget, taTrainParam)
    local criterion = taTrainParam.criterion

    local teOutput = mNet:forward(taInput)
    local criterion = nn.MSECriterion()
    local fErr = criterion:forward(teOutput, teTarget)

    return fErr
  end

  function trainerPool.trainSparseInputNet(mNet, taInput, teTarget, nMaxIteration, strOptimMethod, isEarlyStop)
		strOptimMethod = strOptimMethod or "SGD"
    local criterion = nn.MSECriterion()
    local taTrainParam = trainerPool.getDefaultTrainParams(teTarget:size(1), strOptimMethod, nMaxIteration )

    local errPrev = math.huge
    local errCurr = math.huge

    for i=1, taTrainParam.nMaxIteration do
      errCurr = trainerPool.pri_trainSparseInputNet_SingleRound(mNet, taInput, teTarget, taTrainParam)

--      --[[
      if isEarlyStop and (errPrev <= errCurr or myUtil.isNan(errCurr))  then
        print("** early stop **")
        return errPrev
			end

      if errCurr ~= nil then
        local message = errCurr < errPrev and "<" or "!>"
        myUtil.log(message, false, taTrainParam.isLog)
        myUtil.log(errCurr, false, taTrainParam.isLog)
        errPrev = errCurr
      else
        error("invalid value for errCurr!")
      end
      --]]

    end

    return errCurr
  end


  return trainerPool
end
