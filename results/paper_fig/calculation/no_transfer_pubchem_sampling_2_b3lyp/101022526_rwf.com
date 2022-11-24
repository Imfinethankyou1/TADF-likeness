%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/101022526_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/101022526_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.9480926923        0.5806128036        1.3163144503
O                 5.1576078922       -0.5659209270        1.5839590086
C                 3.8707065371       -0.5853351580        1.1288196876
C                 3.1389843464       -1.7477746720        1.4172457883
C                 1.8269205070       -1.8606166899        0.9892887921
C                 1.1885163709       -0.8257966172        0.2763145253
C                 1.9365673708        0.3288578730       -0.0214257251
C                 3.2661707874        0.4436133662        0.4020085922
C                 1.2776789593        1.4434405142       -0.7997809897
C                 0.3636843524        0.8704738962       -1.8947171361
C                -0.7029622869       -0.1039552852       -1.3482859787
C                -0.2193550947       -0.9285710092       -0.1579778071
C                -1.1297717974       -1.7305371734        0.4255913171
C                -2.5331391383       -1.8007248031       -0.1118040834
C                -3.1165282373       -0.3906558143       -0.3924841033
C                -4.4122423007       -0.5026176784       -1.2186056554
C                -3.4474212544        0.2874606091        0.9389168510
O                -4.1847106777       -0.1183852918        1.8276022915
C                -2.7632024698        1.5865319186        0.9926939620
O                -2.9628118541        2.3964720061        2.0509342817
C                -2.0180371625        1.7704790853       -0.1119787571
C                -2.0788958144        0.5833599233       -1.0495019874
H                 5.5144352267        1.4842463207        1.7649807592
H                 6.9221059680        0.3854013009        1.7689231673
H                 6.0757236987        0.7410794258        0.2373757491
H                 3.6277635205       -2.5500426456        1.9610784752
H                 1.2863837848       -2.7804895154        1.1918465738
H                 3.8155627910        1.3461448984        0.1553625719
H                 2.0385953545        2.0931895010       -1.2487463378
H                 0.6968652093        2.0753863808       -0.1144964642
H                 0.9962873999        0.3359278677       -2.6140657016
H                -0.1215478428        1.6818200468       -2.4527056226
H                -0.9274166101       -0.8177607636       -2.1563526892
H                -0.8790812495       -2.3269409466        1.2987784432
H                -2.5563645394       -2.3749423957       -1.0524223955
H                -3.2000873932       -2.3165000214        0.5868365063
H                -4.8420458108        0.4853415434       -1.4203399334
H                -4.2145395519       -0.9906869343       -2.1798219036
H                -5.1583685014       -1.0925468615       -0.6756895693
H                -3.5717465417        1.9027461015        2.6378935714
H                -1.4498855013        2.6674780418       -0.3319801154
H                -2.4584891391        0.9207357792       -2.0256334091


