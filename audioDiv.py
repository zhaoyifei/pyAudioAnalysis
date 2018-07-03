import audioSegmentation_noGTK as aS
import numpy as np
import os
import sys


def get_filePath_fileName_fileExt(filename):
    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)
    return filepath,shotname,extension


def audioSeg(audio_path):
    [flagsInd, classesAll, acc, CM] = aS.mtFileClassification(audio_path, "data/svmSM", "svm", True)
    # [flagsInd, classesAll, acc, CM] = aS.hmmSegmentation('/Users/zhaoyifei/Downloads/fanghua0305.wav', 'hmmcount', True)				# test 2
    flags = [classesAll[int(f)] for f in flagsInd]
    (segs, classes) = aS.flags2segs(flags, 1)
    #
    # # print classes
    # print '---------------'
    # print len(segs)
    # print '---------------'

    # for i in range(len(segs)):
    #     print classes[i], segs[i][0], segs[i][1]
    [filepath,shotname,extension] = get_filePath_fileName_fileExt(audio_path)
    outputFile = filepath+"/"+shotname+"-audio-seg.csv"
    print outputFile
    with open(outputFile, 'w') as f:
        for i in range(len(segs)):
            f.writelines(classes[i]+", "+np.string_(segs[i][0])+", "+np.string_(segs[i][1])+"\n")


def main(audio_path):
    audioSeg(audio_path)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('Usage: main.py <aideo_file>')
        exit(1)
    main(sys.argv[1])


# import audioSegmentation as aS
# # aS.trainHMM_fromFile('radioFinal/train/bbc4A.wav', 'radioFinal/train/bbc4A.segments', 'hmmTemp1', 1.0, 1.0)	# train using a single file
# # aS.trainHMM_fromDir('radioFinal/small/', 'hmmTemp2', 1.0, 1.0)							# train using a set of files in a folder
# aS.hmmSegmentation('data/scottish.wav', 'hmmTemp1', True, 'data/scottish.segments')				# test 1

