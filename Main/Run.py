from Main import Pre_processing,read
import Proposed_HFGSO_DRN.run
import ResNet.run
import DCNN.DCNN_run
import Focal_Net.Focal_net
import Panoptic_model.Panoptic

def callmain(dts,tr_p):
    acc,sen,spe=[],[],[]
    Pre_processing.process()
    Feat = read.read_data()
    Label = read.read_label()
    ################### Calling Methods #################
    print("\n Proposed HFGSO-based DRN..")
    Proposed_HFGSO_DRN.run.classify(Feat,Label,tr_p,acc,sen,spe)
    ResNet.run.classify(Feat,Label,tr_p,acc,sen,spe)
    Focal_Net.Focal_net.callmain(Feat,Label,tr_p,acc,sen,spe)
    Panoptic_model.Panoptic.classify(Feat,Label,tr_p,acc,sen,spe)
    print("\n Please Wait..")
    DCNN.DCNN_run.callmain(Feat,Label,tr_p,acc,sen,spe)
    print("\n Done.")
    return acc,sen,spe

