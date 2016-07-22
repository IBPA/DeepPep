local lSettings = {}

do
  lSettings.strBaseDir = "/home/ameen/mygithub/depos/DeepPep/sparseData"
  lSettings.strFilenameTarget = string.format("%s/target.csv", lSettings.strBaseDir)
  lSettings.strFilenameMetaInfo = string.format("%s/metaInfo.csv", lSettings.strBaseDir)
  lSettings.nRows=400

  return lSettings
end

