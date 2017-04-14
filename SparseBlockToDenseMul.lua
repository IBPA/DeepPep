
local SparseBlockToDenseMul, parent = torch.class('nn.SparseBlockToDenseMul', 'nn.Module')

function SparseBlockToDenseMul:__init(dMul, dBias)
	self.dMul = dMul or 1
	self.dBias = dBias or 0
end

function SparseBlockToDenseMul:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = torch.ones(input.nBatchSize, 1)
	-- the following only done to support matrix operation only accumulation of results (instead of iterating using a for loop), hence maybe avoided
	self.outputBufferA = torch.ones(input.nBatchSize, 1) -- used for initial output result of each column (allocated for maximum possible size)
	self.outputBufferB = torch.ones(input.nBatchSize, 1) -- used for holding "scatter" results, to be added to self.output
end

function SparseBlockToDenseMul:pri_updateOutput_column(taInput)
	-- copy input to buffer (in right places)
	local nRows = taInput.teValue:size(1)
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, 1)
	self.outputBufferB:scatter(1, teDstIdx, taInput.teValue)

	-- add buffer to output
	self.output:cmul(self.outputBufferB)

	-- cleanup the buffer
	self.outputBufferB:scatter(1, teDstIdx, 1)

end

function SparseBlockToDenseMul:updateOutput(input)
	self:pri_ensureOutput(input)
	self.output:fill(1)

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i])
	end

	self.output = self.output * self.dMul + self.dBias

	return self.output
end

function SparseBlockToDenseMul:pri_ensureGradInput(input)
	if self.gradInput ~= nil then
		return
	end

	self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taGradInputCurr = { teValue = torch.zeros(taInputCurr.teValue:size()),
												teRowIdx = taInputCurr.teRowIdx }

		table.insert(self.gradInput.taData, taGradInputCurr)
	end
end

function SparseBlockToDenseMul:pri_updateGradInput_column(taInput, teGradOutput, taGradInput)

	-- copy teGradOutput to teGradOutputSelected based on teRowIdx
	local nRows = taInput.teValue:size(1)
	local teGradOutputSelected = self.outputBufferA:narrow(1, 1, nRows)
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, 1)
	teGradOutputSelected:gather(teGradOutput, 1, teDstIdx)

	-- copy self.output to taGradInput.teValue based on teRowIdx
	taGradInput.teValue:gather(self.output, 1, teDstIdx)

	-- find the rows with "zero" and assign "1" to them (this is to avoid

	-- devide the output by input of this column (which is actually the gradient with respect to this input), then multiply by gradient
	dEpsilon = 1e-300
	taGradInput.teValue:cdiv(taInput.teValue + dEpsilon) -- to avoid devision by zero, the better solution is to somehow the calculation for these records
	taGradInput.teValue:cmul(teGradOutputSelected)
	taGradInput.teValue:mul(self.dMul)

	-- cleanup teGradOutputSelected
	teGradOutputSelected:fill(1)
end

function SparseBlockToDenseMul:updateGradInput(input, gradOutput)
	self:pri_ensureGradInput(input)

	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput,
																		self.gradInput.taData[i])
	end

	return self.gradInput
end

