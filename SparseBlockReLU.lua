local SparseBlockReLU, parent = torch.class('nn.SparseBlockReLU', 'nn.Module')

-- Module description:
-- strict limitted support assuming: a) inplace update, b) parallel inputs with shared default value, c) fullbatch, d) no bias in prev layers which enables for sparse backpropagaion.
function SparseBlockReLU:__init(isInplace, isFullGradInput)
	self.isInplace = isInplace or true
	self.isFullGradInput = isFullGradInput or false
	self.output = {}
	self.gradInput = {}
end

local function pri_ApplyTensor(x)
	if x <= 0 then
		return 0
	end
end

local function pri_ApplyTensor2(x, y)
	if y <= 0 then
		return 0
	end
end

function SparseBlockReLU:updateOutput(input)
	assert(self.isInplace, "only supporting inplace for now")

	-- update default:
	if input.teDefault ~= nil then
		input.teDefault:apply(pri_ApplyTensor)
	end

	-- update data
	for key, taBlockSparse in pairs(input.taData) do
		taBlockSparse.teValue:apply(pri_ApplyTensor)
	end

	self.output = input

	return self.output
end

function SparseBlockReLU:updateGradInput(input, gradOutput)
	assert(self.isInplace, "only supporting inplace for now")
	assert(self.isFullGradInput == false, "only supporting sparse gradOutput")

	-- update default
	if input.teDefault ~= nil then
		gradOutput.teDefault:map(input.teDefault, pri_ApplyTensor2)
	end

	-- update data
	local nColumns = table.getn(gradOutput.taData)
	for i=1, nColumns do
			local taInputCurr = input.taData[i]
			local taGradOutputCurr = gradOutput.taData[i]
			taGradOutputCurr.teValue:map(taInputCurr.teValue, pri_ApplyTensor2)
	end
	
	self.gradInput = gradOutput

	return self.gradInput
end
