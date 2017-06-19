torch.manualSeed(1)
require('./CExperimentSparseBlock.lua')
require('./CDataLoader.lua')
local dataLoad = dataLoad or require('../MyCommon/dataLoad.lua')

function getExprSetting(strFilePathSetting)
	local taSetting = dataLoad.loadTaSetting(strFilePathSetting)
	taSetting.nRows = tonumber(taSetting.nRows)
	taSetting.nArchId = tonumber(taSetting.nArchId)
  taSetting.nOutputFrameConv1 = tonumber(taSetting.nOutputFrameConv1)
  taSetting.nWindowSizeConv1 = tonumber(taSetting.nWindowSizeConv1)
  taSetting.nWindowSizeMaxPool1 = tonumber(taSetting.nWindowSizeMaxPool1)

	return taSetting
end

local archFactory = require('./deposArchFactory.lua')
local trainerPool = trainerPool or require('./deposTrainerPool.lua')

-- 1) initialize
print("=== trainPred settings: ===")
local strFilePathSetting = arg[1]
local taSetting = getExprSetting(strFilePathSetting)
print(taSetting)

local fuArchBuilder = archFactory.getArchBuilder(taSetting.nArchId)

local oDataLoader = CDataLoader.new(taSetting)
oExperiment = CExperimentSparseBlock.new(oDataLoader, fuArchBuilder)
oExperiment:buildArch(taSetting)

-- 2) train
oExperiment:roundTrip()
local taTrainParams = trainerPool.getDefaultTrainParams(nil, "RMSprop", 10)
taTrainParams.taOptimParams.learningRate = 0.01
oExperiment:train(taTrainParams)

-- 3) predict
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

