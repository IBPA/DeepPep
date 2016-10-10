do
	local deposUtil = {}

	function deposUtil.printBlockSparseInput(taInput)
		local taData = taInput.taData
		for i=1, #taData do
			
			io.write(string.format("#%d: ", i))
			print(taData[i].teValue:squeeze(3))

		end
	end

	return deposUtil
end
