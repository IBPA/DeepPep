require('./CExperimentSparseBlock.lua')


CExperimentSparseBlockFlex, parent = torch.class("CExperimentSparseBlockFlex", "CExperimentSparseBlock" )

function CExperimentSparseBlockFlex:__init(oDataLoader, fuArchBuilder)
	parent.__init(self, oDataLoader)
	self.fuArchBuilder = fuArchBuilder
end

function CExperimentSparseBlockFlex:buildArch()
  self.taMetaInfo = self.oDataLoader:loadSparseMetaInfo()
	self.fuArchBuilder(self)
end

function CExperimentSparseBlockFlex:train(taTrainParams)
	parent.train(self, 20, "SGD", false, nil, taTrainParams)
end
