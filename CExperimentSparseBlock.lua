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
require './SparseBlockSum.lua'

local trainerPool = require('./deposTrainerPool.lua')
local deposUtil = deposUtil or require('./deposUtil.lua')

CExperimentSparseBlock = torch.class("CExperimentSparseBlock")


function CExperimentSparseBlock:__init(oDataLoader)
  self.oDataLoader = oDataLoader
end

function CExperimentSparseBlock:buildArch_Linear(dDropoutRate)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
  dDropoutRate = dDropoutRate or 0.6

	self.mFirst = nn.Sequential()
		self.mFirst:add(nn.SparseBlockFlattenDim3())
		self.mFirst:add(nn.SparseBlockLinear(1, true))
--		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

	self.mRest = nn.Sequential()
		self.mRest:add(nn.SparseBlockToDenseLinear(1, true))
--		self.mRest:add(nn.Sigmoid())

	self.mNet = nn.Sequential()
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
--		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

		self.mFirst:add(nn.SparseBlockTemporalConvolution(5, 10, 8))
		self.mFirst:add(nn.SparseBlockReLU())
		self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize))
--		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

		self.mFirst:add(nn.SparseBlockFlattenDim3())
		self.mFirst:add(nn.SparseBlockLinear(1, false))

	self.mRest = nn.SparseBlockToDenseLinear(1, false)

	self.mNet:add(self.mFirst)
	self.mNet:add(self.mRest)
end

-- ******************************************************************************************************
-- ********* Method to allow transfer of parameters between CExperiment and CExperimentSparseBlock ******
-- ******************************************************************************************************

function CExperimentSparseBlock:setModelParameters(nLayerId, nColumnId, teWeight, teBias)
	if nLayerId == 1 then
		local mCurrent = self.mNet.modules[1].modules[2]
		assert(mCurrent.__typename == "nn.SparseBlockLinear", "only works for nn.SparseBlockLinear!")

		mCurrent:pri_getSubWeight(nColumnId):copy(teWeight)
		mCurrent:pri_getBias(nColumnId):copy(teBias)

	elseif nLayerId == 2 then
		local mCurrent = self.mNet.modules[2].modules[1]
		assert(mCurrent.__typename == "nn.SparseBlockToDenseLinear", "only support nn.SparseBlockToDenseLinear here!")

		mCurrent.weight:copy(teWeight)
		mCurrent.bias:copy(teBias)
	else
		error("not here!")
	end
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

function CExperimentSparseBlock:test()
  -- 1) load input
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()


	-- 3) forward
	local teOutput = self.mNet:forward(taInput)

	print(teOutput)

	--[[
	local taParametersA, taParametersB, taParametersC, taParametersD = self.mNet:getParameters()
	print(taParametersA:size())
	print(taParametersB:size())
	--]]

	

--	print(teOutput)

end

function CExperimentSparseBlock:train(nIteration, strOptimMethod, isEarlyStop, dStopError, taTrainParam)
  local nIteration = nIteration or 20

  -- 1) load input
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)

  -- 2) Load the Target
  local teTarget = self.oDataLoader:loadTarget()

  -- 3) Train
	----[[
	self.mNet:training()
  sys.tic()
  local dTrainErr = trainerPool.trainSparseInputNet(self.mNet, taInput, teTarget, nIteration, strOptimMethod, isEarlyStop, dStopError, taTrainParam)
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


function CExperimentSparseBlock:getConfidenceOne(teOutputAll, taOutputFirst, taFirstProt, taInput)
  -- 1) save the prot column we are about the replace
	local teFirstProtValueOrig = taFirstProt.teValue:clone()
	
	-- 2) replace the protein column with teDefault/zero
	if taFirstProt.teDefault ~= nil then
		local teTmpView = taFirstProt.teDefault:view(1, taFirstProt.teDefault:size(1))
		taFirstProt.teValue:copy(teTmpView:expand(taFirstProt.teValue:size()))
	else
		taFirstProt.teValue:zero()
	end

  -- 3) calculate the final prediction
	local teOutputAllNew = self.mRest:forward(taOutputFirst):clone()

  -- 4) calculate the difference
  local teOutputResidual = torch.add(torch.mul(teOutputAllNew, -1),
                                     teOutputAll):abs():squeeze()
	

  -- 5) replace the orig column
	taFirstProt.teValue:copy(teFirstProtValueOrig)

  -- 6) calculate prot_pepdide confidences
	local dSum = self:getNormalizedResidualSum(taInput, teOutputResidual)
	return dSum /teOutputAll:size(1)
end


function CExperimentSparseBlock:getConfidenceOneVFast(teOutputAll, taOutputFirst, taFirstProt, taInput, nProtId)
  -- 1) save the prot column we are about the replace
	local teFirstProtValueOrig = taFirstProt.teValue:clone()
	
  -- 3) calculate the columns contribution in in final prediction
    self.mRest:pub_setColIds(nProtId)
	local teOutputResidual = self.mRest:forward(taOutputFirst):clone():abs():squeeze()
	
  -- 6) calculate prot_pepdide confidences
	local dSum = self:getNormalizedResidualSum(taInput, teOutputResidual)
	return dSum /teOutputAll:size(1)
end

function CExperimentSparseBlock:getNormalizedResidualSum(taInput, teOutputResidual)
	local dSum = 0
	local nMatchingPeptides = taInput.teRowIdx:size(1)
	for i=1, nMatchingPeptides do
		local n_ji = taInput.teValue[i]:sum()
		local nIdx = taInput.teRowIdx[i]:squeeze()
		local c_ij = teOutputResidual[nIdx]/n_ji
		dSum = dSum + c_ij
	end

	return dSum
end

function CExperimentSparseBlock:getConfidenceRange()
  sys.tic()
	self.mNet:evaluate()
  
	local nEnd = #self.taMetaInfo
	local taInput = self.oDataLoader:loadSparseBlockInput(self.taMetaInfo)
  
  local teOutputAll = self.mNet:forward(taInput):clone()

	local taOutputFirst = self.mFirst:forward(taInput) 


  local taProtInfo = {}
	for i=1, nEnd do
		local dConf = self:getConfidenceOneVFast(teOutputAll, taOutputFirst, taOutputFirst.taData[i], taInput.taData[i], i)
		local strProtFilename = self.taMetaInfo[i].strFilename
    local strProtName = strProtFilename:sub(1, strProtFilename:len() -4)  -- remove the ".txt" from the end
    local taRow = { strProtName, dConf }
    table.insert(taProtInfo, taRow)
	end

  print("confidence total elapsed time(s):" .. sys.toc())
  return taProtInfo
end

function CExperimentSparseBlock:saveResult(taProtInfo)
  self.oDataLoader:saveProtInfo(taProtInfo)
	self.oDataLoader:saveModelParams(self.mNet:getParameters())

	if self.strArchDescription ~= nil then
		self.oDataLoader:saveDescription(self.strArchDescription)
	end

end

