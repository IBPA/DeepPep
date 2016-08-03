local SparseLinearX, parent = torch.class('nn.SparseLinearX', 'nn.Module')

function SparseLinearX:__init(inputSize, outputSize)
   parent.__init(self)

   -- Todo: remove artifitial initial weights and implement reset with random
   self.weight = torch.Tensor(outputSize, inputSize):fill(1)
   self.bias = torch.Tensor(outputSize, 1):fill(0.5)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize, 1)
end


function SparseLinearX:updateOutput(input)
  local output = torch.zeros(input.nBatchSize, self.bias:size(1)) -- ToDo: just using dimention applicable to our particular problem

  local nRecords = input.teOnes:size(1)
  for i=1, nRecords do
    local nRowId = input.teOnes[i][1]
    local nStartId = input.teOnes[i][2]
    local nLength = input.teOnes[i][3]

    output[nRowId]:add(self.weight:narrow(2, nStartId, nLength):sum(2))
  end

  output:add(self.bias:expandAs(output))

  return output
end
