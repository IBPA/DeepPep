-- Description: takes SparseBlock columns (with ) and perfoms a Linear 
-- 							operation to map them into a dense matrix which it's 
-- 							width is specified as nOutputWidth


local SparseBlockToDenseLinear, parent = torch.class('nn.SparseBlockToDenseLinear', 'nn.Module')

function SparseBlockToDenseLinear:__init(nOutputWidth, bias, stdv)
	self.bias = bias or false
	self.stdv = stdv
	self.nOutputWidth = nOutputWidth
end

function SparseBlockToDenseLinear:pri_ensureWeight(input)
	if self.weight ~= nil then
		return
	end

	local nColumns = table.getn(input.taData)
	self.weightMeta = torch.LongTensor(nColumns)

	-- find the size:
	local nTotalWeightSize = 0
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		local nWidth = taInputCurr.teValue:size(2)
		self.weightMeta[i] = nTotalWeightSize + 1
		nTotalWeightSize = nTotalWeightSize + nWidth
	end

	-- allocate:
	self.weight = torch.zeros(nTotalWeightSize, self.nOutputWidth)
	self.gradWeight = torch.zeros(nTotalWeightSize, self.nOutputWidth)

	if self.bias then
		self.bias = torch.zeros(1, self.nOutputWidth)
		self.gradBias = torch.zeros(1, self.nOutputWidth)
	end

	self:reset(self.stdv)
end

function SparseBlockToDenseLinear:reset(stdv)

		if stdv then
			stdv = stdv * math.sqrt(3)
		else
			stdv = 1./math.sqrt(self.weight:size(1))
		end

		self.weight:uniform(-stdv, stdv)
		
		if self.bias then
			self.bias:uniform(-stdv, stdv)
		end
end

function SparseBlockToDenseLinear:pri_getSubW(i, teW)
	local nStart = self.weightMeta[i]
	local nLenght = -1  
	if i < self.weightMeta:size(1) then
		nLenght = self.weightMeta[i+1] - nStart
	else -- the last one, is different
		nLenght = teW:size(1) - nStart + 1
	end

	return teW:narrow(1, nStart, nLenght)
end

function SparseBlockToDenseLinear:pri_getSubWeight(i)
	return self:pri_getSubW(i, self.weight)
end

function SparseBlockToDenseLinear:pri_getSubGradWeight(i)
	return self:pri_getSubW(i, self.gradWeight)
end

function SparseBlockToDenseLinear:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = torch.zeros(input.nBatchSize, self.nOutputWidth)
	-- the following only done to support matrix operation only accumulation of results (instead of iterating using a for loop), hence maybe avoided
	self.outputBufferA = torch.zeros(input.nBatchSize, self.nOutputWidth) -- used for initial output result of each column (allocated for maximum possible size)
	self.outputBufferB = torch.zeros(input.nBatchSize, self.nOutputWidth) -- used for holding "scatter" results, to be added to self.output
end

function SparseBlockToDenseLinear:pri_updateOutput_column(taInput, teWeight)
--	self.outputBufferA:zero() -- no need to reset all to zero, only what's used here
--	self.outputBufferB:zero() -- ToDo: possible optimization: instead of this, can scatter zero scalar to what's non-zero at the end
	local nInputWidth = taInput.teValue:size(2)

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
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, self.nOutputWidth)
	self.outputBufferB:scatter(1, teDstIdx, teOutput)

	-- add buffer to output
	self.output:add(self.outputBufferB)

	-- cleanup the buffer
	self.outputBufferB:scatter(1, teDstIdx, 0)
end

function SparseBlockToDenseLinear:pub_setColIds(nColId)
    self.nPubColId = nColId
end

-- Description: pri_getColIds: enables calculating a given column only.
function SparseBlockToDenseLinear:pri_getColIds(input)
  if self.nPubColId then
    local taR = {}
    taR[self.nPubColId] = "this very value should not be read"
    return taR
  end
  
  return input.taData
end

function SparseBlockToDenseLinear:updateOutput(input)
	self:pri_ensureWeight(input)
	self:pri_ensureOutput(input)
	self.output:zero()

	if self.bias then
		local teBiasExpanded = self.bias:expand(self.output:size(1), self.nOutputWidth)
		self.output:add(teBiasExpanded)
	end

  local taCols = self:pri_getColIds(input)
	for i, _ in pairs(taCols) do
		self:pri_updateOutput_column(input.taData[i], 
																 self:pri_getSubWeight(i))
	end

	return self.output
end

function SparseBlockToDenseLinear:pri_ensureGradInput(input)
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

function SparseBlockToDenseLinear:pri_updateGradInput_column(taInput, teGradOutput, taGradInput, teWeight)

	-- copy teGradOutput to teGradOutputSelected based on teRowIdx
	local nRows = taInput.teValue:size(1)
	local teGradOutputSelected = self.outputBufferA:narrow(1, 1, nRows)
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, self.nOutputWidth)
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

function SparseBlockToDenseLinear:updateGradInput(input, gradOutput)
	self:pri_ensureGradInput(input)

	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput,
																		self.gradInput.taData[i],
																		self:pri_getSubWeight(i))
	end

	return self.gradInput
end

function SparseBlockToDenseLinear:pri_accGradWeight_column(taInput, teGradOutput, teGradWeight, scale)
	-- copy teGradOutput to teGradOutputSelected based on teRowIdx
	local nRows = taInput.teValue:size(1)
	local teGradOutputSelected = self.outputBufferA:narrow(1, 1, nRows)
	local teDstIdx = torch.expand(taInput.teRowIdx, nRows, self.nOutputWidth)
	teGradOutputSelected:gather(teGradOutput, 1, teDstIdx)

  teGradWeight:t():addmm(scale, teGradOutputSelected:t(), taInput.teValue)

	if taInput.teDefault then
		-- a) use taInput.teDefault as if, 100 sparse
		local nInputWidth = taInput.teDefault:size(1)
		local teDefaultInputExpanded = taInput.teDefault:view(1, nInputWidth):expand(teGradOutput:size(1), nInputWidth)
		teGradWeight:t():addmm(scale, teGradOutput:t(), teDefaultInputExpanded)

		-- b) substrcat back extra teDefaults items added
		teDefaultInputExpanded = taInput.teDefault:view(1, nInputWidth):expand(nRows, nInputWidth)
  	teGradWeight:t():addmm(-scale, teGradOutputSelected:t(), teDefaultInputExpanded)
	end

	if self.bias then
		self.gradBias:add(scale, teGradOutput:sum(1))
	end

	-- cleanup teGradOutputSelected
	teGradOutputSelected:zero()
end

function SparseBlockToDenseLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		self:pri_accGradWeight_column(input.taData[i],
																	gradOutput,
																	self:pri_getSubGradWeight(i), scale)
	end

end
