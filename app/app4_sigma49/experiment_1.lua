require('../../CExperiment.lua')
require '../../CDataLoader.lua'

local exprSetting = require('./lSettings.lua')
local oDataLoader = CDataLoader.new(exprSetting)
local oExperiment = CExperiment.new(oDataLoader)

oExperiment:buildArch()
oExperiment:train(3)

