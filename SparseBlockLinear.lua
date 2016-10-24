local SparseBlockLinear, parent = torch.class('nn.SparseBlockLinear', 'nn.Module')

function SparseBlockLinear:__init(nOutputPerColumn, bias)
	bias = bias or false
	assert(bias == false, "Only supporting zero bias for now!")

	self.nOutputPerColumn = nOutputPerColumn
end

function SparseBlockLinear:pri_ensureWeight(input)
	local taWeight = {}

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		local nWidth = taInputCurr.teValue:size(2)
		teWeightCurr = torch.Tensor(self.nOutputPerColumn, nWidth)
		table.insert(taWeight, teWeightCurr)
	end
end

function SparseBlockLinear:pri_ensureOutput(input)
	assert(false, "not implemented")
end

function SparseBlockLinear:updateOutput(input)
	self:pri_ensureWeight(input)
	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i])
	end

	return self.output
end

function SparseBlockLinear:updateGradInput(input, gradOutput)
	assert(false, "not implemented")

	return self.gradInput
end

function SparseBlockLinear:accGradParameters(input, gradOutput, scale)
	assert(false, "not implemented")

end
