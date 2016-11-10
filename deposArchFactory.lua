do
	local deposArchFactory = {}

	deposArchFactory.taBuilders = {
		function(self) -- 1
			self.strArchDescription = "MaxPooling(nPoolingWindow=4), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 2
			self.strArchDescription = "MaxPooling(nPoolingWindow=8), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(8))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 3
			self.strArchDescription = "MaxPooling(nPoolingWindow=12), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(12))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 4
			self.strArchDescription = "MaxPooling(nPoolingWindow=16), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(16))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 5
			self.strArchDescription = "MaxPooling(nPoolingWindow=20), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(20))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 6
			self.strArchDescription = "MaxPooling(nPoolingWindow=24), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(24))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 7
			self.strArchDescription = "MaxPooling(nPoolingWindow=28), Linear(nNodes=1)"

			self.mFirst = nn.Sequential()
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(28))

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 8
			self.strArchDescription = "Linear(nNodes=1)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(1, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 9
			self.strArchDescription = "Linear(nNodes=2)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 10
			self.strArchDescription = "Linear(nNodes=3)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(3, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 11
			self.strArchDescription = "Conv(in=1, out=5, kW=8), ReLU, MaxPool(w=4), Conv(in=5, out=10, kW=8), ReLU, MaxPool(w=3), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 5, 8))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

				self.mFirst:add(nn.SparseBlockTemporalConvolution(5, 10, 8))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 12
			self.strArchDescription = "Conv(in=1, out=2, kW=8), ReLU, MaxPool(w=4), Conv(in=2, out=4 kW=8), ReLU, MaxPool(w=3), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 2, 8))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

				self.mFirst:add(nn.SparseBlockTemporalConvolution(2, 4, 8))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 13
			self.strArchDescription = "Conv(in=1, out=1, kW=8), ReLU, MaxPool(w=4), Conv(in=1, out=1 kW=8), ReLU, MaxPool(w=3), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 8))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

				self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 8))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
		function(self) -- 14
			self.strArchDescription = "Conv(in=1, out=1, kW=4), ReLU, MaxPool(w=4), Conv(in=1, out=1 kW=4), ReLU, MaxPool(w=3), Linear(nNodes=2)"

			self.mFirst = nn.Sequential()

				self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 4))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(4))

				self.mFirst:add(nn.SparseBlockTemporalConvolution(1, 1, 4))
				self.mFirst:add(nn.SparseBlockReLU())
				self.mFirst:add(nn.SparseBlockTemporalMaxPooling(3))


				self.mFirst:add(nn.SparseBlockFlattenDim3())
				self.mFirst:add(nn.SparseBlockLinear(2, false))

			self.mRest = nn.SparseBlockToDenseLinear(1, false)

			self.mNet = nn.Sequential()
				self.mNet:add(self.mFirst)
				self.mNet:add(self.mRest)
		end,
	}

	function deposArchFactory.getArchBuilder(nArchId)
		return deposArchFactory.taBuilders[nArchId]
	end

	return deposArchFactory
end
