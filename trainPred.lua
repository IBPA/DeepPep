--[[ Description:
        To train nn and predicts protein identification probabilities]]

torch.manualSeed(1)
require('./CExperiment.lua')
require('./CData.lua')
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

local archFactory = require('./archFactory.lua')
local trainerLib = trainerLib or require('./trainerLib.lua')

-- 1) initialize
print("=== trainPred settings: ===")
local strFilePathSetting = arg[1]
local taSetting = getExprSetting(strFilePathSetting)
print(taSetting)

local fuArchBuilder = archFactory.getArchBuilder(taSetting.nArchId)

local oData = CData.new(taSetting)
oExperiment = CExperiment.new(oData, fuArchBuilder)
oExperiment:buildArch(taSetting)

-- 2) train
oExperiment:roundTrip()
local taTrainParams = trainerLib.getDefaultTrainParams(nil, "RMSprop", 10)
taTrainParams.taOptimParams.learningRate = 0.01
oExperiment:train(taTrainParams)

-- 3) predict
local taProtInfo = oExperiment:getConfidenceAll()
oExperiment:saveResult(taProtInfo)
