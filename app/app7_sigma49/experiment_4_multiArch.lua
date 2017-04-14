torch.manualSeed(1)
require('../../CDataLoader4.lua')
require('../../CExperimentSparseBlockFlex_Data4.lua')

local exprSetting = require('./lSettings.lua')
local archFactory = require('../../deposArchFactory.lua')
local trainerPool = trainerPool or require('../../deposTrainerPool.lua')

-- 1) initialize
local nExprId = tonumber(arg[1])
exprSetting.setExprId(nExprId)
local fuArchBuilder = archFactory.getArchBuilder(nExprId)

local oDataLoader = CDataLoader4.new(exprSetting, false) -- true if using depectabilities
oExperiment = CExperimentSparseBlockFlex_Data4.new(oDataLoader, fuArchBuilder)
oExperiment:buildArch()

-- 2) train
oExperiment:roundTrip()


---[[
local taTrainParams = trainerPool.getDefaultTrainParams(nil, "SGD", 800)
taTrainParams.taOptimParams.learningRate = 2.00
--taTrainParams.taOptimParams.coefL2 = 0.001
oExperiment:train(taTrainParams)

-- 3) predict
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

--]]
