require 'nn'

require('../../CDataLoader.lua')
local exprSetting = require('./lSettings.lua')

local cDataLoader = CDataLoader.new(exprSetting)

local taX_Sparse = cDataLoader:loadSparseInput()
--local teTarget = cDataLoader:loadTarget()
--print(teTarget)

----[[
table.insert(taX_Sparse, torch.Tensor({{1, 0}}))
local module = nn.SparseLinear(exprSetting.nInputWidth, 1, true)
local teOut = module:forward(taX_Sparse)
print(teOut)
--]]
