require('../../CExperiment.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

local isRetrain = true
local oExperiment
local exprSetting = require('./lSettings.lua')

if isRetrain then
  local oDataLoader = CDataLoader.new(exprSetting)
  oExperiment = CExperiment.new(oDataLoader)

  oExperiment:buildArch()
  oExperiment:train(20, true)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)

else
  oExperiment = CExperiment.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end


local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

