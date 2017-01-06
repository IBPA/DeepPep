local lSettings = {}

do
  lSettings.strBaseDir = "/Users/ameen/mygithub/depos/app/app9_yeast_pl/sparseData3"
  lSettings.strFilenameTarget = string.format("%s/target.csv", lSettings.strBaseDir)
  lSettings.strFilenameMetaInfo = string.format("%s/metaInfo.csv", lSettings.strBaseDir)
  lSettings.strFilenameProtRef = string.format("%s/yeast_reference.csv", lSettings.strBaseDir)
  lSettings.strFilenameProtInfo = string.format("%s/protInfo.csv", lSettings.strBaseDir)
  lSettings.nRows=4670

	function lSettings.setExprId(nExprId)
		lSettings.nExprId = nExprId
 	  lSettings.strFilenameProtInfo = string.format("%s/protInfo_expr_%d.csv", lSettings.strBaseDir, nExprId)
 	  lSettings.strFilenameExprDescription = lSettings.strFilenameProtInfo .. ".desc"	
 	  lSettings.strFilenameExprParams = lSettings.strFilenameProtInfo .. ".params"	
	end

  return lSettings
end

