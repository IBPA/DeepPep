local SparseBlockFlattenDim3, parent = torch.class('nn.SparseBlockFlattenDim3', 'nn.Module')

function SparseBlockFlattenDim3:__init()
end

function SparseBlockFlattenDim3:pri_ensureOutput(input)
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

function SparseBlockFlattenDim3:pri_updateOutput_column(input, output)
	local nRows = input.teValue:size(1)
	torch.view(output.teValue, input.teValue, nRows, -1)
end

function SparseBlockFlattenDim3:updateOutput(input)
	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i])
	end
	
	return self.output
end

function SparseBlockFlattenDim3:pri_ensureGradInput(input)
	if self.gradInput ~= nil then
		return
	end

	self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taGradInputCurr = { teValue = torch.Tensor(),
												teRowIdx = taInputCurr.teRowIdx }
		table.insert(self.gradInput.taData, taGradInputCurr)
	end
end

function SparseBlockFlattenDim3:pri_updateGradInput_column(taInput, taGradOutput, taGradInput)
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue
	local gradInput = taGradInput.teValue
	torch.viewAs(gradInput, gradOutput, input)
end

function SparseBlockFlattenDim3:updateGradInput(input, gradOutput)
	self:pri_ensureGradInput(input)

	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput.taData[i],
																		self.gradInput.taData[i])
																		
	end
	
	return self.gradInput
end
