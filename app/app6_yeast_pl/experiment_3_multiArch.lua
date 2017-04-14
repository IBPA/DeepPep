--[[
	variations:
	1) number of convolutional[+dropout,maxpooling,relu] layers:
		a: dropout rate, b: nChannels(input/output), c: nKwSize, d: nPoolingWindowSize
--]]

torch.manualSeed(1)
require('../../CExperimentSparseBlockFlex.lua')
require('../../CDataLoader.lua')

local exprSetting = require('./lSettings.lua')
local archFactory = require('../../deposArchFactory.lua')
local trainerPool = trainerPool or require('../../deposTrainerPool.lua')

-- 1) initialize
local nExprId = tonumber(arg[1])
exprSetting.setExprId(nExprId)
local fuArchBuilder = archFactory.getArchBuilder(nExprId)

local oDataLoader = CDataLoader.new(exprSetting)
oExperiment = CExperimentSparseBlockFlex.new(oDataLoader, fuArchBuilder)
oExperiment:buildArch()

-- 2) train
oExperiment:roundTrip()

----[[

local taTrainParams = trainerPool.getDefaultTrainParams(nil, "SGD", 10)
taTrainParams.taOptimParams.learningRate = 10.0
oExperiment:train(taTrainParams)

taTrainParams = trainerPool.getDefaultTrainParams(nil, "SGD", 500)
taTrainParams.taOptimParams.learningRate = 1.0
oExperiment:train(taTrainParams)

-- 3) predict
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)


--]]
