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

function sparseLinearX_test.updateGradInput()
  local nInputWidth = 10
  local nOutputWidth = 1
  local taInputCSparse = { nBatchSize = 10,
                                                  --rowid, startId, length
                           teOnes = torch.LongTensor({{1, 3, 5},
                                                      {2, 1, 3},
                                                      {2, 7, 3},
                                                      {8, 8, 2}} )}
  local mSparseLinearX = nn.SparseLinearX(nInputWidth, nOutputWidth)
  local teOutput = mSparseLinearX:forward(taInputCSparse)

  local gradOutput = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}})
  local teGradInput = mSparseLinearX:updateGradInput(taInputCSparse, gradOutput)
  print(teGradInput)
end

function sparseLinearX_test.accGradParameters()
  local nInputWidth = 10
  local nOutputWidth = 1
  local taInputCSparse = { nBatchSize = 10,
                                                  --rowid, startId, length
                           teOnes = torch.LongTensor({{1, 3, 5},
                                                      {2, 1, 3},
                                                      {2, 7, 3},
                                                      {8, 8, 2}} )}
  local mSparseLinearX = nn.SparseLinearX(nInputWidth, nOutputWidth)
  local teOutput = mSparseLinearX:forward(taInputCSparse)

  local gradOutput = torch.Tensor({{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}})
  mSparseLinearX:accGradParameters(taInputCSparse, gradOutput)
  print(mSparseLinearX.gradWeight)

end


--wrapUT(sparseLinearX_test.updateOutput, "updateOutput")
--wrapUT(sparseLinearX_test.updateGradInput, "updateGradInput")
wrapUT(sparseLinearX_test.accGradParameters, "accGradParameters")
