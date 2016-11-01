require('../../CExperiment.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

local dDropout = tonumber(arg[1])
local isRetrain = true
local oExperiment
local exprSetting = require('./lSettings.lua')


--[[
  local oDataLoader = CDataLoader.new(exprSetting)
  oExperiment = CExperiment.new(oDataLoader)

  oExperiment:buildArch(dDropout)

	oExperiment:test()
	--]]
----[[
if isRetrain then
  local oDataLoader = CDataLoader.new(exprSetting)
  oExperiment = CExperiment.new(oDataLoader)

  oExperiment:buildArch(dDropout)
  oExperiment:train(30, true)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)

else
  oExperiment = CExperiment.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end


----[[ToDo: uncomment
local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)
--]]

--]]
