--[[
Plan:
	1) load sparse data into sparseBlockTensor
	2) implement ReLU sparseBlockTensor in, sparseBlockTensor out
	3) implement TemporalConv sparseBlockTensor in, sparseBlockTensor out
	4) implement TemporalConv with shared weights + async queue

--]]

require 'nn'
require('../SparseBlockReLU.lua')
require '../SparseBlockTemporalConvolution.lua'
local deposUtil = deposUtil or require('../deposUtil.lua')

local sparseBlockTensor_test = {}

local taInput1 = { teDefault = torch.Tensor(1, 1, 1):fill(0),
									 nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({2, 4}),
										 	 teValue = torch.Tensor({{{0, 0, 1, 1, 1, 1, 0},
											 												{1, 1, 1, -10, 0, 0, 0}}}) },
										 { teRowIdx = torch.LongTensor({2}),
										 	 teValue = torch.Tensor({{{ 0, -0.1, 1 }}}) }
										}
									}

local taInput2 = { nBatchSize = 4,
									 taData = {
										 { teRowIdx = torch.LongTensor({2, 4}),
										 	 teValue = torch.Tensor({{{0}, {0}, {1}, {1}, {1}, {1}, {0}},
											 												{{1}, {1}, {1}, {2}, {0}, {0}, {0}}}) },
										 { teRowIdx = torch.LongTensor({2}),
										 	 teValue = torch.Tensor({ {{0}, {-0.1}, {1} }}) }
										}
									}



function sparseBlockTensor_test.ReLU_test1()
	local mNet = nn.SparseBlockReLU()
	local taOutput = mNet:forward(taInput2)
	deposUtil.printSparseBlockInput(taOutput)
end

function sparseBlockTensor_test.TemporalConvolution_test1()
	local mnet = nn.SparseBlockTemporalConvolution(1, 1, 3, 1)
	print("input***:")
	deposUtil.printSparseBlockInput(taInput2)

	local taOutput = mnet:forward(taInput2)
	print("output***:")
	deposUtil.printSparseBlockInput(taOutput)

end

--sparseBlockTensor_test.ReLU_test1()
sparseBlockTensor_test.TemporalConvolution_test1()
