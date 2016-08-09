require('../../CExperiment.lua')
require '../../CDataLoader.lua'

local isRetrain = false
local oExperiment
local exprSetting = require('./lSettings.lua')

if isRetrain then
  local oDataLoader = CDataLoader.new(exprSetting)
  oExperiment = CExperiment.new(oDataLoader)

  oExperiment:buildArch()
  oExperiment:train(1, true)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)

else
  oExperiment = CExperiment.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end


local taMetaInfo = oExperiment:getConfidenceRange()
print(taMetaInfo)

