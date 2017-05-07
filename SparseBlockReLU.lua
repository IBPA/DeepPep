local SparseBlockReLU, parent = torch.class('nn.SparseBlockReLU', 'nn.Module')

-- Module description:
-- strict limitted support assuming: a) inplace update, b) parallel inputs with shared default value, c) fullbatch, d) no bias in prev layers which enables for sparse backpropagaion.
function SparseBlockReLU:__init(isInplace, isFullGradInput, dMin, dMax)
	self.isInplace = isInplace or true
	self.isFullGradInput = isFullGradInput or false
  self.dMin = dMin or 0
  self.dMax = dMax or math.huge
	self.output = {}
	self.gradInput = {}
  
  
  self.fuApplyTensor = function(x)
    if x <= self.dMin then
      return self.dMin
    end
    
    if x >= self.dMax then
      return self.dMax
    end
  end
  
  self.fuApplyTensor2 = function(x, y)
    if y <= self.dMin then
      return self.dMin
    end
    
    if y >= self.dMax then
      return self.dMax
    end
  end
  
end




function SparseBlockReLU:updateOutput(input)
	assert(self.isInplace, "only supporting inplace for now")

	-- update default:
	if input.teDefault ~= nil then
		input.teDefault:apply(self.fuApplyTensor)
	end

	-- update data
	for key, taBlockSparse in pairs(input.taData) do
		taBlockSparse.teValue:apply(self.fuApplyTensor)
	end

	self.output = input

	return self.output
end

function SparseBlockReLU:updateGradInput(input, gradOutput)
	assert(self.isInplace, "only supporting inplace for now")
	assert(self.isFullGradInput == false, "only supporting sparse gradOutput")

	-- update default
	if input.teDefault ~= nil then
		gradOutput.teDefault:map(input.teDefault, self.fuApplyTensor2)
	end

	-- update data
	local nColumns = table.getn(gradOutput.taData)
	for i=1, nColumns do
			local taInputCurr = input.taData[i]
			local taGradOutputCurr = gradOutput.taData[i]
			taGradOutputCurr.teValue:map(taInputCurr.teValue, self.fuApplyTensor2)
	end
	
	self.gradInput = gradOutput

	return self.gradInput
end
