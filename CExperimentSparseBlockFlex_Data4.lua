require('./CExperimentSparseBlockFlex.lua')


CExperimentSparseBlockFlex_Data4, parent_Data4 = torch.class("CExperimentSparseBlockFlex_Data4", "CExperimentSparseBlockFlex" )

function CExperimentSparseBlockFlex_Data4:__init(oDataLoader, fuArchBuilder)
	parent_Data4.__init(self, oDataLoader)
	self.fuArchBuilder = fuArchBuilder
end

function CExperimentSparseBlockFlex_Data4:getNormalizedResidualSum(taInput, teOutputResidual)
	local dSum = 0
	local nMatchingPeptides = taInput.teIdx:size(1)
	for i=1, nMatchingPeptides do
		local nIdx = taInput.teIdx[i][1]
		local c_ij = teOutputResidual[nIdx]
		dSum = dSum + c_ij
	end

	return dSum
end


