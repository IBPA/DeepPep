local SparseBlockSum, parent = torch.class('nn.SparseBlockSum', 'nn.Module')

function SparseBlockSum:__init()
end

function SparseBlockSum:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taOutputCurr = { teValue = torch.Tensor(),
										 teRowIdx = taInputCurr.teRowIdx }

		table.insert(self.output.taData, taOutputCurr)
	end
end

function SparseBlockSum:pri_ensureTaIndices(input)
	if self.taIndices ~= nil then
		return
	end

	self.taIndices = {}

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		table.insert(self.taIndices, taInputCurr.teValue.new())
	end
end

function SparseBlockSum:pri_updateOutput_column(taInput, taOutput, teIndices)
	local input = taInput.teValue
	local output = taOutput.teValue

	local nDim1 = input:size(1)
	local nDim3 = input:size(3)

	
	output:resize(nDim1, 1, nDim3)

	output:copy(torch.sum(input, 2))

end

function SparseBlockSum:updateOutput(input)
	self:pri_ensureTaIndices(input)
	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i],
																 self.taIndices[i])
	end
	
	return self.output
end

function SparseBlockSum:updateGradInput(input, gradOutput)
	-- just supporting first layer for now, no need to bp
	return gradOutput
end

function SparseBlockSum:empty()
   assert(false, "empty not implemented!")
end

