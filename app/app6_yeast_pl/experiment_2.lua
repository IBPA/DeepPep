require('../../CExperimentSparseBlock.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

torch.setdefaulttensortype('torch.FloatTensor')
local exprSetting = require('./lSettings.lua')
local oExperiment
local isRetrain = true


if isRetrain then
	local oDataLoader = CDataLoader.new(exprSetting)
	oExperiment = CExperimentSparseBlock.new(oDataLoader)

	oExperiment:buildArch(dDropout, 3)
	oExperiment:roundTrip()
	oExperiment:train(500, "SGD", false, 0.001)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)
else
	oExperiment = CExperimentSparseBlock.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end

local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

