%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/139344429_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/139344429_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -4.3382431442       -1.1787457337        1.3478234587
C                -4.4421874600       -0.3297436548        0.4735848725
C                -5.5332546446        0.7237510176        0.5214355315
C                -6.2080975727        0.9635409209        1.8843901619
C                -5.3075242602        1.9745336288        1.3030552551
F                -5.8040145269        3.1395811140        0.8236563671
F                -4.0828741478        2.1955297197        1.8447424653
N                -3.5961904423       -0.2861094440       -0.6129993303
C                -3.4210618925        0.8973726309       -1.4518555007
C                -2.1540739416        1.6624020907       -1.0463563386
N                -1.0325697377        0.7300864498       -0.9451673828
C                 0.1877631928        1.1449417952       -0.4834574917
C                 0.4178591228        2.4874924281       -0.0838331271
C                 1.6884648486        2.7843814233        0.3690499348
N                 2.6863861682        1.8901589145        0.4420143835
C                 2.3730781534        0.6452555778        0.0366757706
C                 3.4051282847       -0.3801998516        0.0823200587
C                 3.3339635020       -1.7202975606       -0.2886303140
N                 4.4993151597       -2.3807900568       -0.1087913976
C                 5.3414980699       -1.4699172740        0.3834753963
C                 6.6954145640       -1.6005757620        0.7594214196
C                 7.3726369512       -0.5122227462        1.2477799289
C                 6.7044529019        0.7392597736        1.3711647397
C                 7.4604835991        1.9191836542        1.9065874564
F                 8.5397087659        2.2022468069        1.1395303617
F                 6.7031786567        3.0340825713        1.9668464277
F                 7.9302743907        1.6791865305        3.1533456587
C                 5.3909468481        0.8776154511        1.0096649170
N                 4.7186584667       -0.2131144402        0.5223606830
N                 1.1770827874        0.2392276820       -0.4202845432
C                -1.2751005965       -0.6419240722       -1.3788104192
C                -2.4802646719       -1.2618518470       -0.6350386028
C                -2.8747094254       -2.5765659650       -1.2555558225
C                -2.1764119509       -3.8090837868       -1.1775879397
N                -2.7731788445       -4.7740393376       -1.8741281496
N                -3.8545909772       -4.1689898911       -2.4063202921
C                -3.9595076414       -2.8579810154       -2.0665606862
H                -6.1579740401        0.8264213911       -0.3627114741
H                -7.2731404884        1.1721554247        1.8931536931
H                -5.8252025770        0.3495630696        2.6939720599
H                -3.3707293265        0.5937936040       -2.5034148736
H                -4.2801379561        1.5599030178       -1.3510788335
H                -2.3324651023        2.1422924163       -0.0761647834
H                -1.9500797926        2.4541794227       -1.7838083260
H                -0.3472062240        3.2514085418       -0.1236555798
H                 1.9302900429        3.7953774242        0.6928892445
H                 2.4571635563       -2.2143606645       -0.6837718304
H                 7.1606690261       -2.5730476370        0.6473341587
H                 8.4136980537       -0.5928938063        1.5420845710
H                 4.8188151288        1.7916854187        1.0736142611
H                -1.4412858980       -0.6794140580       -2.4639665643
H                -0.3802801164       -1.2241525022       -1.1679158776
H                -2.1931821135       -1.4311532296        0.4076207200
H                -1.2671889201       -4.0298647271       -0.6326254484
H                -4.4901996585       -4.7156443594       -2.9675619428
H                -4.7840041493       -2.2439254887       -2.3980590853


