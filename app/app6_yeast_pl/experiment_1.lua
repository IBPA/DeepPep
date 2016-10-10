require('../../CExperiment.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

local dDropout = tonumber(arg[1])
local isRetrain = true
local oExperiment
local exprSetting = require('./lSettings.lua')

if isRetrain then
  local oDataLoader = CDataLoader.new(exprSetting)
  oExperiment = CExperiment.new(oDataLoader)

  oExperiment:buildArch(dDropout)
  oExperiment:train(1, true) --30
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)

else
  oExperiment = CExperiment.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end


----[[ToDo: uncomment
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)
--]]

