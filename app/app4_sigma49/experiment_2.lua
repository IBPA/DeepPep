require('../../CExperimentSparseBlock.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

local exprSetting = require('./lSettings.lua')

local oDataLoader = CDataLoader.new(exprSetting)
local oExperiment = CExperimentSparseBlock.new(oDataLoader)

oExperiment:buildArch(dDropout)
oExperiment:train(30, true)
io.read()

--local taProtInfo = oExperiment:getConfidenceRange()
--oExperiment:saveResult(taProtInfo)

