require 'nn'

require('../../CDataLoader.lua')
local exprSetting = require('./lSettings.lua')

local cDataLoader = CDataLoader.new(exprSetting)

local taX_Sparse = cDataLoader:loadSparseInputSingle("p0.sprs")
print(taX_Sparse)

local nWidth=393
