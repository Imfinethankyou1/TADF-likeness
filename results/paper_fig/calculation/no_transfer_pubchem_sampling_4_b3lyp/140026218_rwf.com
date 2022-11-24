%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/140026218_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/140026218_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.1441684516        4.2040644080        1.0302264379
C                 3.7823792375        4.3662162194        0.3437814231
C                 3.7521264704        3.8853931938       -1.1192495330
C                 3.9212243818        2.3697968642       -1.3311160327
C                 2.7736153684        1.4597449750       -0.8951370135
C                 1.5218756653        1.9943587374       -0.6038669542
C                 0.4586023840        1.1628862111       -0.2782975201
C                 0.5574668013       -0.2252338055       -0.3165112970
C                 1.8220055418       -0.8014630882       -0.6171023976
C                 2.0172930364       -2.3109069897       -0.7076691461
C                 2.2569793255       -2.8411113077       -2.1415453817
C                 2.3365376320       -4.3721978950       -2.2017231115
C                 2.5982037084       -4.9057136551       -3.6140426922
C                 2.9367205652        0.0502785163       -0.8395118348
C                 4.3537673114       -0.5058559521       -0.9593629289
C                 5.2009670391       -0.3618254461        0.3319093892
C                 4.7621988102       -1.2527338716        1.5010645784
C                 5.6124040217       -1.0389569135        2.7581274783
C                -0.7330130398       -0.8482787316        0.2927280375
C                -1.3002914536       -2.1784847104       -0.3022795917
C                -1.2424399333       -2.4670079142       -1.8215293756
C                -2.4266647756       -3.3087348184       -2.3303141279
C                -2.5062321900       -4.7132927957       -1.7190791371
C                -0.3118114689       -1.1451776106        1.7918863578
C                -1.1192013900       -2.1400104030        2.6565531102
C                -0.7102262141       -2.0621668361        4.1352385609
C                -1.4168079601       -3.1017346469        5.0111720197
C                -1.8182015158        0.3896086884        0.1866858438
C                -2.8511333323        0.3369501571        1.3493012684
C                -3.6537283948        1.5980876113        1.7310444679
C                -4.6893622372        1.2909224846        2.8242625239
C                -5.4693896111        2.5301608010        3.2753140748
C                -2.5387648469        0.3208578172       -1.1929730034
C                -3.7674114536        1.2063567371       -1.4883988728
C                -4.1851053385        1.0988999105       -2.9629477022
C                -5.4526705669        1.8956387883       -3.2881394485
C                -0.8581063692        1.6271812249        0.2920865473
N                -1.2837994366        2.9378932250       -0.2372675866
C                -1.4201006699        4.0591518748        0.5999793887
O                -1.3416490590        4.0590564847        1.8127998020
C                -1.6493964070        5.2438638991       -0.2886494396
C                -1.5651983414        4.8397639268       -1.5574735501
C                -1.2797101227        3.3646426897       -1.5790060187
O                -1.0729387753        2.6890794408       -2.5649979401
H                 5.1213746243        4.6135394833        2.0465629494
H                 5.4418159791        3.1522734217        1.1105394895
H                 5.9322878947        4.7302515045        0.4767092648
H                 3.5028861393        5.4284322368        0.3634161491
H                 3.0178870730        3.8387282907        0.9276710048
H                 4.5654738779        4.3852430501       -1.6638192869
H                 2.8267143844        4.2269813797       -1.6003678391
H                 4.0880577740        2.1989441819       -2.4059531179
H                 4.8538176652        2.0609139418       -0.8507575922
H                 1.3752509444        3.0687387260       -0.6013439998
H                 1.1475036494       -2.8230116363       -0.2938968375
H                 2.8597788897       -2.6209725697       -0.0784501108
H                 3.1819010827       -2.4204606712       -2.5529735235
H                 1.4592552688       -2.4836271430       -2.8026520790
H                 1.4011364691       -4.8028564134       -1.8157230186
H                 3.1299430967       -4.7209310201       -1.5248618629
H                 2.6496515174       -6.0004859617       -3.6255355840
H                 3.5458395352       -4.5234167561       -4.0130531338
H                 1.8030440966       -4.6019953136       -4.3058969203
H                 4.8801227435        0.0032060282       -1.7751818570
H                 4.3493256952       -1.5607041457       -1.2340351287
H                 6.2463167007       -0.5943596483        0.0817566262
H                 5.1924801963        0.6824328752        0.6698472735
H                 3.7080199926       -1.0558634805        1.7356387245
H                 4.8222071301       -2.3079489935        1.1984626343
H                 6.6710926816       -1.2515045346        2.5636736527
H                 5.2870241331       -1.6910634774        3.5767499756
H                 5.5430628804       -0.0025332193        3.1108475286
H                -0.8116285827       -3.0152695941        0.2049975970
H                -2.3452324890       -2.2415335652        0.0309332979
H                -0.3263763694       -3.0191072702       -2.0454068692
H                -1.1809573479       -1.5440662718       -2.4026730648
H                -3.3689143787       -2.7757020938       -2.1368248194
H                -2.3485344847       -3.3990442524       -3.4222432229
H                -3.3440542146       -5.2813210133       -2.1396233216
H                -1.5879029098       -5.2812512822       -1.9164860978
H                -2.6458022094       -4.6777398930       -0.6324142643
H                 0.7134204183       -1.5328126624        1.7577958382
H                -0.2283624701       -0.1913337196        2.3286403588
H                -0.9454189628       -3.1644861578        2.3036046855
H                -2.1991494364       -1.9766259475        2.5772953907
H                 0.3784668803       -2.1915595235        4.2183388019
H                -0.9244255262       -1.0541218372        4.5183125651
H                -2.5062303909       -2.9786094206        4.9722755309
H                -1.1880680370       -4.1220111340        4.6791090568
H                -1.1097152558       -3.0164202746        6.0598477954
H                -3.5605381199       -0.4754702369        1.1324323469
H                -2.3377702948        0.0471164417        2.2679456496
H                -4.1722052411        2.0346205112        0.8734553501
H                -2.9710387394        2.3670180571        2.1078433774
H                -5.3925447924        0.5284752765        2.4584502715
H                -4.1821234382        0.8440323884        3.6916491153
H                -4.7956897881        3.2964190869        3.6775437073
H                -6.0180247916        2.9794381226        2.4380184045
H                -6.1985925267        2.2829361110        4.0555942917
H                -2.9013071086       -0.7006238705       -1.3222348800
H                -1.7976464744        0.4932658377       -1.9786597832
H                -3.5909989796        2.2602807170       -1.2495358975
H                -4.6128205098        0.8907155847       -0.8632877472
H                -3.3571102283        1.4474352045       -3.5937028057
H                -4.3433700076        0.0411561774       -3.2191731598
H                -6.3072759337        1.5461630656       -2.6952468617
H                -5.3167683662        2.9629062330       -3.0718530243
H                -5.7226904775        1.8042304286       -4.3465935314
H                -0.7124914928        1.8452279019        1.3571263447
H                -1.8291210309        6.2318571037        0.1155457677
H                -1.6594358554        5.4079031406       -2.4741352474


