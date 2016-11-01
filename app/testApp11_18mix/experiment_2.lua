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

	oExperiment:buildArch(dDropout, 4)
	oExperiment:roundTrip()
	--oExperiment:train(400, "SGD", false)
	oExperiment:train(200, "SGD", false)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)
else
	oExperiment = CExperimentSparseBlock.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end

local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

