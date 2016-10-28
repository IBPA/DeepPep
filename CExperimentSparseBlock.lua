require 'nn'
require 'sys'

--ToDo: implement:
require('./SparseBlockReLU.lua')
require './SparseBlockTemporalConvolution.lua'
require './SparseBlockTemporalMaxPooling.lua'
require './SparseBlockFlattenDim3.lua'
require './SparseBlockLinear.lua'
require './SparseBlockToDenseLinear.lua'
require './SparseBlockDropout.lua'

local trainerPool = require('./deposTrainerPool.lua')
local deposUtil = deposUtil or require('./deposUtil.lua')

CExperimentSparseBlock = torch.class("CExperimentSparseBlock")

function CExperimentSparseBlock:__init(oDataLoader)
  self.oDataLoader = oDataLoader
end

function CExperimentSparseBlock:buildArch_Linear(dDropoutRate)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
  dDropoutRate = dDropoutRate or 0.6

	self.mNet = nn.Sequential()
		self.mFirst = nn.Sequential()
		self.mFirst:add(nn.SparseBlockFlattenDim3())
		self.mFirst:add(nn.SparseBlockLinear(1))
		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

	self.mRest = nn.Sequential()
		self.mRest:add(nn.SparseBlockToDenseLinear(1))
		self.mRest:add(nn.Sigmoid())

	self.mNet:add(self.mFirst)
	self.mNet:add(self.mRest)

end

function CExperimentSparseBlock:buildArch(dDropoutRate, nPoolWindowSize)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
  dDropoutRate = dDropoutRate or 0.2
	nPoolWindowSize = nPoolWindowSize or 4

	self.mNet = nn.Sequential()

	self.mFirst = nn.Sequential()
		self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 5, 8))
		self.mFirst:add(nn.SparseBlockReLU())
		self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize))
		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

		self.mFirst:add(nn.SparseBlockTemporalConvolution(5, 10, 8))
		self.mFirst:add(nn.SparseBlockReLU())
		self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize))
		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

		self.mFirst:add(nn.SparseBlockFlattenDim3())
		self.mFirst:add(nn.SparseBlockLinear(1))

	self.mRest = nn.SparseBlockToDenseLinear(1)

	self.mNet:add(self.mFirst)
	self.mNet:add(self.mRest)
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

function CExperimentSparseBlock:train(nIteration, strOptimMethod, isEarlyStop)
  local nIteration = nIteration or 20

  -- 1) load input
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()

  -- 3) Train
	----[[
	self.mNet:training()
  sys.tic()
  local dTrainErr = trainerPool.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration, strOptimMethod, isEarlyStop)
  print("\ntraining error:" .. dTrainErr) 
  print("training elapsed time(s):" .. sys.toc())
	--]]
end

function CExperimentSparseBlock:save(strFilePath)
  torch.save(strFilePath, self)
end

function CExperimentSparseBlock.loadFromFile(strFilePath)
  local oExperiment = torch.load(strFilePath)

  return oExperiment
end

function CExperimentSparseBlock:getConfidenceOne(teOutputAll, taOutputFirst, teProtFirst, taInput)
  -- 1) save the prot column we are about the replace
  local teProtOrig = teProtFirst:clone()
	
	-- 2) replace the protein column with zero (works since no bias)
	teProtFirst:zero()

  -- 3) calculate the final prediction
	local teOutputAllNew = self.mRest:forward(taOutputFirst)

  -- 4) calculate the difference
  local teOutputResidual = torch.add(torch.mul(teOutputAllNew, -1),
                                     teOutputAll):abs():squeeze()
	

  -- 5) replace the orig column
	teProtFirst:copy(teProtOrig)

  -- 6) calculate prot_pepdide confidences
	local dSum = 0
	local nMatchingPeptides = taInput.teRowIdx:size(1)
	for i=1, nMatchingPeptides do
		local n_ji = taInput.teValue:sum()
		local c_ij = math.abs(teOutputResidual[i])/n_ji
		dSum = dSum + c_ij
	end

	return dSum/teOutputAll:size(1)
end

function CExperimentSparseBlock:getConfidenceRange()
  sys.tic()
	self.mNet:evaluate()

	local nEnd = #self.taMetaInfo
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)
  local teOutputAll = self.mNet:forward(taInput):clone()
	local taOutputFirst = self.mFirst.output -- no need to recalculate

  local taProtInfo = {}
	for i=1, nEnd do
		local dConf = self:getConfidenceOne(teOutputAll, taOutputFirst, taOutputFirst.taData[i].teValue, taInput.taData[i])
		local strProtFilename = self.taMetaInfo[i].strFilename
    local strProtName = strProtFilename:sub(1, strProtFilename:len() -4)  -- remove the ".txt" from the end
    local taRow = { strProtName, dConf }
    table.insert(taProtInfo, taRow)
	end

  print("confidence total elapsed time(s):" .. sys.toc())
  return taProtInfo
end

function CExperimentSparseBlock:saveResult(taProtInfo)
  local taRef = self.oDataLoader:loadProtRef()

  for key, value in pairs(taProtInfo) do
    local strProtName = value[1]

    if taRef[strProtName] == nil then
      table.insert(value, 0)
    else
      table.insert(value, 1)
    end
  end

  self.oDataLoader:saveProtInfo(taProtInfo)
end

