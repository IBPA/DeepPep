require 'nn'
require '../SparseLinearX.lua'

local sparseLinearX_test = {}

function wrapUT(fuUT, strName)
  fuUT()
  print("PASS " .. strName)
end

function sparseLinearX_test.updateOutput()
  local nInputWidth = 10
  local taInputCSparse = { nBatchSize = 10,
                                                  --rowid, startId, length
                           teOnes = torch.LongTensor({{1, 3, 5},
                                                      {2, 1, 3},
                                                      {2, 7, 3},
                                                      {8, 8, 2}} )}
  local mSparseLinearX = nn.SparseLinearX(nInputWidth, 1)
  local teOutput = mSparseLinearX:forward(taInputCSparse)
  print(teOutput)

end

wrapUT(sparseLinearX_test.updateOutput, "updateOutput")
