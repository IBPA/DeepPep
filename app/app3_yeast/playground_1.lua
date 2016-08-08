require 'nn'
require 'sys'
require '../../SparseLinearX.lua'
require '../../CDataLoader.lua'
local exprSetting = require('./lSettings.lua')
local trainerPool = require('../../deposTrainerPool.lua')
torch.manualSeed(1)

local cDataLoader = CDataLoader.new(exprSetting)

--[[
local taX_Sparse0 = cDataLoader:loadSparseInputSingleV2("YAL005C.txt")
print(taX_Sparse0)
local nWidth0=642
local module0 = nn.SparseLinearX(nWidth0, 1)
local output0 = module0:forward(taX_Sparse0)
--print(output0)
--]]

----[[
local taMetaInfo = cDataLoader:loadSparseMetaInfo()

-- 1) Build the layer0
local nUnitWidthLayer0 = 1
local mLayer0 = nn.ParallelTable()
local nParallels = 0
for key, taFileInfo in pairs(taMetaInfo) do
  mLayer0:add(nn.SparseLinearX(taFileInfo.nWidth, nUnitWidthLayer0 ))
  nParallels = nParallels + 1
end
----[[

-- 2) Build the rest of the FNN:
local mNet = nn.Sequential()
mNet:add(mLayer0)
mNet:add(nn.JoinTable(2))
mNet:add(nn.Linear(nParallels, 1))


-- 3) Load the Sparse Input:
local taInput = {}
local counter = 0
for key, taFileInfo in pairs(taMetaInfo) do
  counter = counter + 1

  --io.write(counter .. ", ")
  --io.flush()
  --print("F:" .. taFileInfo.strFilename)

  local taOneInput = cDataLoader:loadSparseInputSingleV2(taFileInfo.strFilename)
  table.insert(taInput, taOneInput)
end

sys.tic()
local output = mNet:forward(taInput)
print("elapsed for single forward: " .. sys.toc())
print(output:size())

-- 4) Load the Target
local teTarget = cDataLoader:loadTarget()

-- 5) Train
----[[
sys.tic()
local dTrainErr = trainerPool.trainSparseInputNet(mNet, taInput, teTarget, 20)
print("\ntraining error:" .. dTrainErr) 
print("training elapsed time(s):" .. sys.toc())
--print("readline:")
--io.read('*line')
  --]]
--]]
