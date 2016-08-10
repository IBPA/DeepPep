require('../../CExperiment.lua')
require '../../CDataLoader.lua'

local isRetrain = false
local oExperiment
local exprSetting = require('./lSettings.lua')

if isRetrain then
  local oDataLoader = CDataLoader.new(exprSetting)
  oExperiment = CExperiment.new(oDataLoader)

  oExperiment:buildArch()
  oExperiment:train(10, true)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)

else
  oExperiment = CExperiment.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end


local taProtConf = oExperiment:getConfidenceRange()
oExperiment:normalizeByMax(taProtConf)
print(taProtConf)
local dAUC = oExperiment:getAUC(taProtConf)

--[[
local taConf = {}
for key, value in pairs(taProtConf) do
  table.insert(taConf, value)
end


print(torch.Tensor(taConf):histc(10))

--]]
