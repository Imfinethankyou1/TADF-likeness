%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/51628816_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/51628816_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.8733611347        3.8153079005       -1.0056452601
C                -0.9092180368        2.3979146324       -1.4585435831
C                -0.7766045600        1.7840547466       -2.6688899158
C                -0.8866648972        0.3742661573       -2.4292489742
C                -1.0800140136        0.2163487416       -1.0879177094
C                -1.2745862933       -0.9950587922       -0.2137936891
C                -0.3964341892       -0.9435610397        1.0592325447
N                 1.0209918631       -0.9012684269        0.7517069269
C                 1.8283747357        0.1425329316        1.0818129566
O                 1.4809268872        1.1403316447        1.7036723697
C                 3.2505380790       -0.0092056765        0.6268700601
C                 4.2975464151        0.9199458875        0.8484983603
C                 5.3837732105        0.3361660419        0.2533017610
C                 6.7687633171        0.7113747643        0.0910271462
C                 7.7658537635        0.0044901300       -0.5447646229
C                 9.0223591438        0.6703772673       -0.5187230945
C                 8.9724080695        1.8731924545        0.1335213340
S                 7.3833011056        2.2190165480        0.7317467624
O                 5.0108284654       -0.8579002032       -0.2851914694
N                 3.6577266921       -1.0675105205       -0.0429833305
N                -2.6599372097       -1.3169028671        0.1616647775
C                -3.3264169300       -0.3331098467        1.0363177709
C                -4.8561796152       -0.4861954152        0.9693660180
N                -5.2075126825       -1.8920362581        0.8471387856
C                -6.6055698543       -2.1764985227        1.1048180779
C                -4.7587199222       -2.4453179452       -0.4346817038
C                -3.5065071718       -1.7151210456       -0.9562747809
O                -1.0999800879        1.4519381337       -0.4862972723
H                -0.6688050454        4.4734098074       -1.8547409342
H                -0.0922156589        3.9673493413       -0.2510787653
H                -1.8269588405        4.1215855070       -0.5569518804
H                -0.6091476829        2.2762664083       -3.6172660607
H                -0.8166587163       -0.4197717975       -3.1607050492
H                -0.9297007024       -1.8487573620       -0.8139729724
H                -0.6096134333       -0.0517766526        1.6503585401
H                -0.6297925531       -1.8277067228        1.6638642762
H                 1.4416546210       -1.6610656733        0.2304673510
H                 4.2259248687        1.8614353237        1.3698757606
H                 7.5920300730       -0.9582316808       -1.0107611669
H                 9.9256035128        0.2710874481       -0.9663812793
H                 9.7737794008        2.5802957922        0.3004953695
H                -3.0669549925        0.7037777053        0.7734613100
H                -2.9941771067       -0.5078614091        2.0666060351
H                -5.2907789544       -0.0911952115        1.8944032434
H                -5.2720943865        0.1264268902        0.1436592886
H                -7.2958770661       -1.7013895684        0.3778360898
H                -6.7724870089       -3.2593118485        1.0621081754
H                -6.8779314383       -1.8323804405        2.1089286382
H                -4.5458895939       -3.5130340360       -0.2925459489
H                -5.5463497982       -2.3689951357       -1.2101905454
H                -3.7994152147       -0.8610908078       -1.5936720588
H                -2.9398264327       -2.3966382991       -1.6011126623


