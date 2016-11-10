require('../../CExperimentSparseBlock.lua')
require '../../CDataLoader.lua'
torch.manualSeed(1)

torch.setdefaulttensortype('torch.FloatTensor')
local exprSetting = require('./lSettings.lua')
local oExperiment
local isRetrain = true


function buildArch(self, nPoolWindowSize)
	nPoolWindowSize = nPoolWindowSize or 4

	self.mNet = nn.Sequential()

	self.mFirst = nn.Sequential()
--		self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 2, 8))
--		self.mFirst:add(nn.SparseBlockReLU())
		self.mFirst:add(nn.SparseBlockTemporalMaxPooling(16))
--		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

--		self.mFirst:add(nn.SparseBlockTemporalConvolution(2, 2, 8))
--		self.mFirst:add(nn.SparseBlockReLU())
--		self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize))
--		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

--		self.mFirst:add(nn.SparseBlockTemporalConvolution(10, 10, 8, 1, true))
--		self.mFirst:add(nn.SparseBlockReLU())
--		self.mFirst:add(nn.SparseBlockTemporalMaxPooling(nPoolWindowSize, 1, true))
--		self.mFirst:add(nn.SparseBlockDropout(dDropoutRate))

		self.mFirst:add(nn.SparseBlockFlattenDim3())
		self.mFirst:add(nn.SparseBlockLinear(2, false))

	self.mRest = nn.SparseBlockToDenseLinear(1, false)

	self.mNet:add(self.mFirst)
	self.mNet:add(self.mRest)
end

if isRetrain then
	local oDataLoader = CDataLoader.new(exprSetting)
	oExperiment = CExperimentSparseBlock.new(oDataLoader)


	oExperiment:buildArch(dDropout, 4)
	buildArch(oExperiment, 4)
	oExperiment:roundTrip()
	oExperiment:train(2000, "SGD", false)
  oExperiment:save(exprSetting.strFilenameExperiment1Obj)
else
	oExperiment = CExperimentSparseBlock.loadFromFile(exprSetting.strFilenameExperiment1Obj)
end

local taProtInfo = oExperiment:getConfidenceRange()
oExperiment:saveResult(taProtInfo)

