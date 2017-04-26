require('./CExperimentSparseBlock.lua')
require('./SparseCleavageProb.lua')
require('./SparseBlockToDenseSum.lua')
require('./SparseBlockToDenseMul.lua')
require('./SparseCleavageProbC.lua')
require('./SparseBlockLinearNonNegativeW.lua')

CExperimentSparseBlockFlex, parent = torch.class("CExperimentSparseBlockFlex", "CExperimentSparseBlock" )

function CExperimentSparseBlockFlex:__init(oDataLoader, fuArchBuilder)
	parent.__init(self, oDataLoader)
	self.fuArchBuilder = fuArchBuilder
end

function CExperimentSparseBlockFlex:buildArch(taArchParams)
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
	self.fuArchBuilder(self, taArchParams)
end

function CExperimentSparseBlockFlex:train(taTrainParams)
	parent.train(self, 20, "SGD", false, 0.0001, taTrainParams)
end

