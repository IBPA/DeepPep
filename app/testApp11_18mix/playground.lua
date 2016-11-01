require('../../CExperiment.lua')
require '../../CDataLoader.lua'
require('../../CExperimentSparseBlock.lua')
local exprSetting = require('./lSettings.lua')

torch.manualSeed(1)

-- Build architecture
local oDataLoader = CDataLoader.new(exprSetting)
local oExperiment = CExperimentSparseBlock.new(oDataLoader)
oExperiment:buildArch_Linear()
oExperiment:roundTrip()

-- loading the old weights into oExperiment
local oldExperiment = CExperiment.loadFromFile(exprSetting.strFilenameExperiment1Obj)

----[[
local nColumns = oldExperiment:getNColumns()
for i=1, nColumns do
	local teWeight, teBias = oldExperiment:getModelParameters(1, i)
	oExperiment:setModelParameters(1, i, teWeight, teBias)
end

local teWeight, teBias = oldExperiment:getModelParameters(2)
oExperiment:setModelParameters(2, nil, teWeight, teBias)
--]]

--local flatParams , flatGradParams = oExperiment.mNet:getParameters()
--flatParams:zero()
--oldExperiment:test()
--oExperiment:test()
-- predict
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)
