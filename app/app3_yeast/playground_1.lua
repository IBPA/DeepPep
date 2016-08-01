require 'nn'
require('../../CDataLoader.lua')
local exprSetting = require('./lSettings.lua')
local trainerPool = require('../../deposTrainerPool.lua')

--[[
local cDataLoader = CDataLoader.new(exprSetting)
local taX_Sparse0 = cDataLoader:loadSparseInputSingle("YAL005C.txt")
local nWidth0=642
local module0 = nn.SparseLinear(nWidth0, 1, true)
local output0 = module0:forward(taX_Sparse0)
--]]
