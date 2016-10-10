local SparseBlockReLU, parent = torch.class('nn.SparseBlockReLU', 'nn.Module')

-- Module description:
-- strict limitted support assuming: a) inplace update, b) parallel inputs with shared default value, c) fullbatch
function SparseBlockReLU:__init(isInplace)
	self.isInplace = isInplace or true
	self.output = {}
	self.gradInput = {}
end

function pri__SparseBlockReLU_ApplyTensor(x)
	if x <= 0 then
		return 0
	end
end

function pri__SparseBlockReLU_ApplyTensor2(x, y)
	if y <= 0 then
		return 0
	end
end

function SparseBlockReLU:updateOutput(input)
	assert(self.isInplace, "only supporting inplace for now")

	-- update default:
	input.teDefault:apply(pri__SparseBlockReLU_ApplyTensor)

	-- update data
	for key, taBlockSparse in pairs(input.taData) do
		taBlockSparse.teValue:apply(pri__SparseBlockReLU_ApplyTensor)
	end

	self.output = input

	return self.output
end

function SparseBlockReLU:updateGradInput(input, gradOutput)
	assert(self.isInplace, "only supporting inplace for now")

	-- update default
	gradOutput.teDefault:map(input.teDefault, pri__SparseBlockReLU_ApplyTensor2)

	-- update data
	local nColumns = table.getn(gradOutput.taData)
	for i=1, nColumns do
			local taInputCurr = input.taData[i]
			local taGradOutputCurr = gradOutput.taData[i]
			taGradOutputCurr.teValue:map(taInputCurr.teValue, pri__SparseBlockReLU_ApplyTensor2)
	end
	
	self.gradInput = gradOutput

	return self.gradInput
end
