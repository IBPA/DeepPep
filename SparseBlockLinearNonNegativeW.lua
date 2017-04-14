-- Description: takes BlockSparse input, and does a linear map on each column from 
--              whatever the length of the column is to the specified length 
--              as nOutputPerColumn
-- Note: currently assumes "default" value in sparse input is "zero"
local SparseBlockLinearNonNegativeW, parent = torch.class('nn.SparseBlockLinearNonNegativeW', 'nn.Module')

function SparseBlockLinearNonNegativeW:__init(nOutputPerColumn, bias)
	self.bias = bias or false

	self.nOutputPerColumn = nOutputPerColumn
end

function SparseBlockLinearNonNegativeW:pri_ensureWeight(input)
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
	self.weight = torch.zeros(nTotalWeightSize, self.nOutputPerColumn)
	self.gradWeight = torch.zeros(nTotalWeightSize, self.nOutputPerColumn)

	if self.bias then
		self.bias = torch.zeros(nColumns, self.nOutputPerColumn)
		self.gradBias = torch.zeros(nColumns, self.nOutputPerColumn)
	end

	self:reset(0.01)
end

function SparseBlockLinearNonNegativeW:pri_getSubW(i, teW)
	local nStart = self.weightMeta[i]
	local nLenght = -1  
	if i < self.weightMeta:size(1) then
		nLenght = self.weightMeta[i+1] - nStart
	else -- the last one, is different
		nLenght = teW:size(1) - nStart + 1
	end

	return teW:narrow(1, nStart, nLenght)
end

function SparseBlockLinearNonNegativeW:pri_getSubWeight(i)
	return torch.clamp(self:pri_getSubW(i, self.weight), 0, 1)
end

function SparseBlockLinearNonNegativeW:pri_getSubGradWeight(i)
	return self:pri_getSubW(i, self.gradWeight)
end

function SparseBlockLinearNonNegativeW:pri_getGradBias(i)
		if not self.bias then
			return nil
		end

		return self.gradBias[i]
end

function SparseBlockLinearNonNegativeW:pri_getBias(i)
		if not self.bias then
			return nil
		end

		return self.bias[i]
end



function SparseBlockLinearNonNegativeW:reset(value)
		self.weight:fill(value)

		if self.bias then
			self.bias:fill(value)
		end

end

function SparseBlockLinearNonNegativeW:pri_ensureOutput(input)
	if self.output ~= nil then
		return
	end

	self.output = { nBatchSize = input.nBatchSize, taData = {} }

	local nColumns = table.getn(input.taData)
	for i=1, nColumns do
		local taInputCurr = input.taData[i]
		local nRows = taInputCurr.teValue:size(1)

		taOutputCurr = { teValue = torch.zeros(nRows, self.nOutputPerColumn),
										 teRowIdx = taInputCurr.teRowIdx}

		if self.bias then
			taOutputCurr.teDefault = torch.Tensor(self.nOutputPerColumn)
		end

		table.insert(self.output.taData, taOutputCurr)
	end

end

function SparseBlockLinearNonNegativeW:pri_updateOutput_column(taInput, taOutput, teWeight, teBias)
	local teInput = taInput.teValue
	local teOutput = taOutput.teValue:fill(0)
	teOutput:addmm(teInput, teWeight)

	if self.bias then
		local teAddBuffer = torch.Tensor(1, teBias:size(1)):copy(teBias):expand(teOutput:size())
		teOutput:add(teAddBuffer)
		taOutput.teDefault:copy(teBias)
	end
end

function SparseBlockLinearNonNegativeW:updateOutput(input)
	self:pri_ensureWeight(input)
	self:pri_ensureOutput(input)

	local nColumns = table.getn(self.output.taData)
	for i=1, nColumns do
		self:pri_updateOutput_column(input.taData[i], 
																 self.output.taData[i],
																 self:pri_getSubWeight(i),
																 self:pri_getBias(i))
	end

	return self.output
end

-- Important Note: 
-- 	Since assumes default sparse value in inut to be zero, therefore backpropagation is only
-- 	needed for non-sparse blocks.
--
function SparseBlockLinearNonNegativeW:pri_ensureGradInput(input)
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

function SparseBlockLinearNonNegativeW:pri_updateGradInput_column(taInput, taGradOutput, taGradInput, teWeight)
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue
	local gradInput = taGradInput.teValue
	gradInput:fill(0)
	gradInput:addmm(gradOutput, teWeight:t())
end

function SparseBlockLinearNonNegativeW:updateGradInput(input, gradOutput)
	self:pri_ensureGradInput(input)

	local nColumns = table.getn(self.gradInput.taData)
	for i=1, nColumns do
		self:pri_updateGradInput_column(input.taData[i],
																		gradOutput.taData[i],
																		self.gradInput.taData[i],
																		self:pri_getSubWeight(i))
	end

	return self.gradInput
end

function SparseBlockLinearNonNegativeW:pri_accGradWeight_column(taInput, taGradOutput, teSubWeight, teGradWeight, teGradBias, scale)
	local input = taInput.teValue
	local gradOutput = taGradOutput.teValue

	local teClampedGrad = teSubWeight -- since teSubWeight itself is a clamped copy, no need to re-copy
	teClampedGrad:apply(function(w)
			if w > 0 and w < 1 then
				return 1
			else
				return 0
			end
		end)

		for i=1, self.nOutputPerColumn do
			local teDgW = teClampedGrad:narrow(2, i, 1):t():expand(input:size())

			local teInputTr = torch.cmul(input, teDgW)


			local teGradOutSlice = gradOutput:narrow(2, i, 1)
--			print(teGradOutSlice)
			teGradWeight:narrow(2, i, 1):t():addmm(scale, teGradOutSlice:t(), teInputTr)
		end


--  teGradWeight:t():addmm(scale, gradOutput:t(), input) --old line

	if teGradBias then
		-- Note: only the sum of all gradOutputs are useful here, hence only that is backpropagated.
		-- 			 i.e. due to zero inputs, gradOutputs with sparse inputs  are not useful anywhere else, and only their vertical sum is useful here.
		teGradBias:add(scale, taGradOutput.teGradOutputSum)
	end
end

function SparseBlockLinearNonNegativeW:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
	local nColumns = table.getn(input.taData)

	for i=1, nColumns do
		self:pri_accGradWeight_column(input.taData[i],
																	gradOutput.taData[i],
																	self:pri_getSubWeight(i),
																	self:pri_getSubGradWeight(i), 
																	self:pri_getGradBias(i), scale)
	end
end
