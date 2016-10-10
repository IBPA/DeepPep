require 'nn'
require 'sys'

--ToDo: implement:
require './SparseBlockReLU.lua'
--require './SparseBlockTemporalConvolution.lua'
--require './SparseBlockTemporalMaxPooling.lua'
--require './SparseBlockDropout.lua'
--require './SparseBlockLinear.lua'

local trainerPool = require('./deposTrainerPool.lua')
local deposUtil = deposUtil or require('./deposUtil.lua')

CExperimentSparseBlock = torch.class("CExperimentSparseBlock")

function CExperimentSparseBlock:__init(oDataLoader)
  self.oDataLoader = oDataLoader
end

function CExperimentSparseBlock:buildArch(dDropoutRate)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
  local dDropoutRate = dDropoutRate or 0.7

	self.mNet = nn.Sequential()
	self.mNet:add(nn.SparseBlockReLU())
end

function CExperimentSparseBlock:train(nIteration)
  local nIteration = nIteration or 20

  -- 1) load input
	local taInput = self.oDataLoader:loadBlockSparseInput(self.taMetaInfo)


  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()


	print("waiting for 20 seconds ...")
	os.execute("sleep 20" )
	print("done")

	deposUtil.printBlockSparseInput(taInput)

  -- 3) Train
	--[[
  sys.tic()
  local dTrainErr = trainerPool.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration)
  print("\ntraining error:" .. dTrainErr) 
  print("training elapsed time(s):" .. sys.toc())
	--]]
end
