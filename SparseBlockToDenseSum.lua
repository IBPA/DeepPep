
local SparseBlockToDenseSum, parent = torch.class('nn.SparseBlockToDenseSum', 'nn.Module')

function SparseBlockToDenseSum:__init()
end

function SparseBlockToDenseSum:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = torch.zeros(input.nBatchSize, 1)
	-- the following only done to support matrix operation only accumulation of results (instead of iterating using a for loop), hence maybe avoided
	self.outputBufferA = torch.zeros(input.nBatchSize, 1) -- used for initial output result of each column (allocated for maximum possible size)
	self.outputBufferB = torch.zeros(input.nBatchSize, 1) -- used for holding "scatter" results, to be added to self.output
end

function SparseBlockToDenseSum:pri_updateOutput_column(taInput)
	local nInputWidth = taInput.teValue:size(2)
	local teWeight = torch.ones(nInputWidth, 1)

	-- calculate output for teDefault input
	if taInput.teDefault then
		local teDefaultInputExpanded = taInput.teDefault:view(1, nInputWidth):expand(self.output:size(1), nInputWidth) -- expand for multiplication

		self.outputBufferB:zero()
		self.outputBufferB:addmm(teDefaultInputExpanded, teWeight) -- so this writes the default, but sparse blocks will be overwritten next
	end

	-- calculate the output for Sparse blocks
	local teInput = taInput.teValue
	local nRows = teInput:size(1)
	local teOutput = self.outputBufferA:narrow(1, 1, nRows)
	teOutput:zero()
	teOutput:addmm(teInput, teWeight)
	
	-- copy result to buffer
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, 1)
	self.outputBufferB:scatter(1, teDstIdx, teOutput)

	-- add buffer to output
	self.output:add(self.outputBufferB)

	-- cleanup the buffer
	self.outputBufferB:scatter(1, teDstIdx, 0)

end

function SparseBlockToDenseSum:updateOutput(input)
	self:pri_ensureOutput(input)
	self.output:zero()

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i])
	end

	return self.output
end

function SparseBlockToDenseSum:pri_ensureGradInput(input)
	if self.gradInput ~= nil then
		return
	end

	self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]

		taGradInputCurr = { teValue = torch.zeros(taInputCurr.teValue:size()),
												teRowIdx = taInputCurr.teRowIdx }

		if taInputCurr.teDefault then
			taGradInputCurr.teGradOutputSum = torch.zeros(1, taInputCurr.teValue:size(2))
		end

		table.insert(self.gradInput.taData, taGradInputCurr)
	end
end

function SparseBlockToDenseSum:pri_updateGradInput_column(taInput, teGradOutput, taGradInput)

	local nWidth = taInput.teValue:size(2)
	local teWeight = torch.ones(nWidth, 1)

	-- copy teGradOutput to teGradOutputSelected based on teRowIdx
	local nRows = taInput.teValue:size(1)
	local teGradOutputSelected = self.outputBufferA:narrow(1, 1, nRows)
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, 1)
	teGradOutputSelected:gather(teGradOutput, 1, teDstIdx)

	-- calculate and update gradInput
	local gradInput = taGradInput.teValue
	gradInput:zero()
	gradInput:addmm(teGradOutputSelected, teWeight:t())

	-- cleanup teGradOutputSelected
	teGradOutputSelected:zero()

	-- calculate gradOutput sum, then multiply by weights (just reordering optimization to save memory)
	if taGradInput.teGradOutputSum then
		local teGradOutputSum = teGradOutput:sum(1)
		taGradInput.teGradOutputSum:mm(teGradOutputSum, teWeight:t())
	end
end


function SparseBlockToDenseSum:updateGradInput(input, gradOutput)
	self:pri_ensureGradInput(input)

	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput,
																		self.gradInput.taData[i])
	end

	return self.gradInput
end

