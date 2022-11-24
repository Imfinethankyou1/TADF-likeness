%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/135094087_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/135094087_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -6.4457298163        1.4832032937        1.3029625627
C                -5.3180818859        1.1608174899        0.3485944200
C                -5.4899345054        1.2926138550       -1.0343514797
C                -4.4499420004        0.9657722342       -1.9057476235
C                -3.2335477305        0.5078540617       -1.4109124024
C                -3.0481739203        0.3743288306       -0.0255908050
N                -1.7947901139       -0.0891568911        0.4186930959
C                -1.3518407614       -0.3209022244        1.6914553890
O                -1.9930685578       -0.1599170893        2.7220868916
C                 0.1181245046       -0.7737313963        1.7459067118
N                 0.6125863566       -1.3648659373        0.5022891943
C                 0.2025238595       -2.7820936412        0.3628280040
C                 1.2399257357       -3.7579927136        0.9287898902
C                 2.5744192082       -3.6197618534        0.1565740000
C                 2.6687634765       -2.2551471151       -0.5574332009
C                 2.0643898610       -1.1199197128        0.2819903651
C                 2.2504472950        0.2632757979       -0.3669184254
C                 3.7030995720        0.7600015111       -0.3683041907
N                 3.8128751596        2.1330462100       -0.8397254468
C                 4.3015220730        2.5801052594       -2.0280077734
C                 4.1452322412        3.9533658793       -2.0353156969
C                 3.5420098251        4.2409998506       -0.7921299717
N                 3.3388839639        3.1385725856       -0.0696360164
C                -4.0927789698        0.7039711835        0.8492925477
H                -7.1664670707        2.1739459997        0.8528345481
H                -6.9951927416        0.5760104504        1.5879654327
H                -6.0706660614        1.9359448705        2.2269871648
H                -6.4366104218        1.6532263241       -1.4291571261
H                -4.5854534909        1.0727357629       -2.9788956434
H                -2.4223445393        0.2589239803       -2.0918600901
H                -1.0965636838       -0.3096299687       -0.2872637709
H                 0.7059912321        0.1219581555        1.9780259277
H                 0.2099616343       -1.4398790555        2.6180581501
H                -0.7634708434       -2.8997550412        0.8634603258
H                 0.0269473281       -3.0038457980       -0.6977617311
H                 1.3902199433       -3.5367011224        1.9931641208
H                 0.8591592455       -4.7847196546        0.8782348030
H                 2.6728319089       -4.4197859791       -0.5866319422
H                 3.4183990320       -3.7350197417        0.8473542503
H                 3.7158901615       -2.0284190580       -0.7861022708
H                 2.1449741476       -2.2897672585       -1.5221016331
H                 2.5901379867       -1.1167333354        1.2558990850
H                 1.6536607202        1.0072637585        0.1716323155
H                 1.8657774706        0.2337514934       -1.3946376402
H                 4.3415340629        0.1491239710       -1.0131426472
H                 4.1179330887        0.7227468740        0.6451466100
H                 4.7190555235        1.8995808038       -2.7570294119
H                 4.4310529304        4.6414621827       -2.8175696517
H                 3.2518490530        5.2027619528       -0.3890835496
H                -3.9355224869        0.6003779659        1.9150353354


