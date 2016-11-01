require('../../CExperimentSparseBlock.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

local exprSetting = require('./lSettings.lua')
local oExperiment
local isRetrain = true

--[[
	local oDataLoader = CDataLoader.new(exprSetting)
	oExperiment = CExperimentSparseBlock.new(oDataLoader)

	oExperiment:buildArch_Linear(0.6)

	oExperiment:test()
--]]

----[[
if isRetrain then
	local oDataLoader = CDataLoader.new(exprSetting)
	oExperiment = CExperimentSparseBlock.new(oDataLoader)

	oExperiment:buildArch_Linear(0.6)

	oExperiment:roundTrip()
	oExperiment:train(200, "SGD", false)
  oExperiment:save(exprSetting.strFilenameExperiment2_LinearObj)
else
	oExperiment = CExperimentSparseBlock.loadFromFile(exprSetting.strFilenameExperiment2_LinearObj)
end

local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

--]]
