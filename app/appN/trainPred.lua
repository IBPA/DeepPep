torch.manualSeed(1)
require('../../CExperimentSparseBlockFlex.lua')
require('../../CDataLoader.lua')
local dataLoad = dataLoad or require('../../../MyCommon/dataLoad.lua')

function getExprSetting(strFilePathSetting)
	local taSetting = dataLoad.loadTaSetting(strFilePathSetting)
	taSetting.nRows = tonumber(taSetting.nRows)
	taSetting.nArchId = tonumber(taSetting.nArchId)

	return taSetting
end

local archFactory = require('../../deposArchFactory.lua')
local trainerPool = trainerPool or require('../../deposTrainerPool.lua')

-- 1) initialize
print("=== trainPred settings: ===")
local strFilePathSetting = arg[1]
local exprSetting = getExprSetting(strFilePathSetting)
print(exprSetting)

local fuArchBuilder = archFactory.getArchBuilder(exprSetting.nArchId)

local oDataLoader = CDataLoader.new(exprSetting)
oExperiment = CExperimentSparseBlockFlex.new(oDataLoader, fuArchBuilder)
oExperiment:buildArch()

-- 2) train
oExperiment:roundTrip()


local taTrainParams = trainerPool.getDefaultTrainParams(nil, "SGD", 200)
taTrainParams.taOptimParams.momentum = 0.5
taTrainParams.taOptimParams.learningRate = 1.0
oExperiment:train(taTrainParams)


-- 3) predict
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

