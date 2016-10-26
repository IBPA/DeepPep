require 'nn'
require 'sys'

--ToDo: implement:
require('./SparseBlockReLU.lua')
require './SparseBlockTemporalConvolution.lua'
require './SparseBlockTemporalMaxPooling.lua'
require './SparseBlockFlattenDim3.lua'
require './SparseBlockLinear.lua'
require './SparseBlockToDenseLinear.lua'

local trainerPool = require('./deposTrainerPool.lua')
local deposUtil = deposUtil or require('./deposUtil.lua')

CExperimentSparseBlock = torch.class("CExperimentSparseBlock")

function CExperimentSparseBlock:__init(oDataLoader)
  self.oDataLoader = oDataLoader
end

function CExperimentSparseBlock:buildArch(dDropoutRate, nPoolWindowSize)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
  dDropoutRate = dDropoutRate or 0.7
	nPoolWindowSize = nPoolWindowSize or 4

	self.mNet = nn.Sequential()


	self.mNet:add(nn.SparseBlockTemporalConvolution(1, 5, 8))
	self.mNet:add(nn.SparseBlockReLU())
	self.mNet:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize))

	self.mNet:add(nn.SparseBlockTemporalConvolution(5, 10, 8))
	self.mNet:add(nn.SparseBlockReLU())
	self.mNet:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize))

	self.mNet:add(nn.SparseBlockFlattenDim3())
	self.mNet:add(nn.SparseBlockLinear(1))
	self.mNet:add(nn.SparseBlockToDenseLinear(1))

end

function CExperimentSparseBlock:roundTrip()
  -- 1) load input
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()


	-- ToDo: now doing simple single forward, backward for until fully implemented
  sys.tic()
	local teOutput = self.mNet:forward(taInput)
  print("forward elapsed time(s):" .. sys.toc())

  sys.tic()
	local taGradInput = self.mNet:backward(taInput, teOutput, 0)
  print("backward elapsed time(s):" .. sys.toc())
end

function CExperimentSparseBlock:train(nIteration)
  local nIteration = nIteration or 20

  -- 1) load input
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()

  -- 3) Train
	----[[
  sys.tic()
  local dTrainErr = trainerPool.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration)
  print("\ntraining error:" .. dTrainErr) 
  print("training elapsed time(s):" .. sys.toc())
	--]]
end
