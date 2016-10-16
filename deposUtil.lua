do
	local deposUtil = {}

	function deposUtil.printSparseBlockInput(taInput)
		local taData = taInput.taData

		if taInput.teDefault ~= nil then
			print("teDefault:" .. tostring(taInput.teDefault:squeeze(3)))
		end

		for i=1, #taData do
			
			io.write(string.format("#%d: ", i))
			print(taData[i].teValue:squeeze(3))

		end
	end

	function deposUtil.mulSparseBlockInput(taInput, dValue)
		taInput.teDefault:mul(dValue)

		local taData = taInput.taData
		for i=1, #taData do
			taData[i].teValue:mul(dValue)
		end
	end

	function deposUtil.getCopyRandomizedBlocks(taInput)
		local taRes = { teDefault = torch.rand(taInput.teDefault:size()),
										taData = {}}

		local taData = taInput.taData
		for i=1, #taData do
			local taNew = { teValue = torch.rand(taData[i].teValue:size()),
											teRowIdx = taData[i].teRowIdx:clone()}
			table.insert(taRes.taData, taNew)
		end


		return taRes
	end

	return deposUtil
end
