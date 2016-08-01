local lSettings = {}

do
  lSettings.strBaseDir = "/home/user/eetemame/mygithub/depos/app/app3_yeast/sparseData"
  lSettings.strFilenameTarget = string.format("%s/target.csv", lSettings.strBaseDir)
  lSettings.strFilenameMetaInfo = string.format("%s/metaInfo.csv", lSettings.strBaseDir)
  --/home/user/eetemame/data/protein/yeast/all/peptideProbs.csv
  lSettings.nRows=11601

  return lSettings
end

