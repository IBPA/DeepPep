local lSettings = {}

do
  lSettings.strBaseDir = "/home/user/eetemame/mygithub/depos/app/app4_sigma49/sparseData2"
  lSettings.strFilenameTarget = string.format("%s/target.csv", lSettings.strBaseDir)
  lSettings.strFilenameMetaInfo = string.format("%s/metaInfo.csv", lSettings.strBaseDir)
  lSettings.nRows=337

  return lSettings
end

