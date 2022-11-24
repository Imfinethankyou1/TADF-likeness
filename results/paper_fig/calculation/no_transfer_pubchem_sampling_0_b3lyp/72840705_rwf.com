%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/72840705_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/72840705_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.5855319917        3.4976642132       -0.4498895170
N                -4.0144970291        2.1867592511       -0.1976681544
C                -3.5339574199        2.0536916380        1.1788277134
C                -3.2446204155        0.5643843138        1.4972962083
N                -2.9262104269       -0.2427268816        0.3203183589
C                -1.6493734534        0.0494273296       -0.3710574434
C                -0.5007025452        0.3115313210        0.6227742068
C                 0.8455050068        0.5343122053       -0.0824058462
N                 1.1496620769       -0.5247919190       -1.0422309645
C                 2.4197056594       -0.9586260923       -1.3473264474
O                 2.6834827030       -1.5857497421       -2.3667127889
C                 3.5154706170       -0.6951803396       -0.3363777048
C                 3.8502447806       -1.6811497649        0.5678239358
C                 4.9261967964       -1.4562192819        1.4608186834
N                 5.6522206386       -0.3570238057        1.4944188488
C                 5.3538158301        0.6213345842        0.5888419992
C                 6.1390298758        1.8047650244        0.5998420512
C                 5.8921120083        2.8186833572       -0.2976048129
C                 4.8512935886        2.6938089206       -1.2504429007
C                 4.0687296522        1.5612430217       -1.2844038829
C                 4.2925368884        0.5004188439       -0.3659559467
C                 0.0886707219       -0.7781700917       -2.0122078979
C                -1.2715467135       -1.0917403656       -1.3610129679
C                -1.4596820578       -2.5626791175       -0.8806942735
C                -1.1063477536       -3.0441402148        0.5408385857
C                 0.3704725838       -3.1531445774        0.9242899573
O                 1.1626394863       -3.8854147880       -0.0081311608
C                -4.1090944372       -0.2469677133       -0.5608474186
C                -4.9076959102        1.0866688674       -0.5397072831
H                -3.8425455185        4.2707878213       -0.2218504458
H                -4.8583674671        3.5883879384       -1.5071819445
H                -5.4876868037        3.7065921505        0.1588617743
H                -2.6304247625        2.6685906498        1.2861192383
H                -4.2721887928        2.4398452036        1.9121564219
H                -4.1376038903        0.1125468148        1.9481428665
H                -2.4557504723        0.4818923977        2.2485885363
H                -1.7461224918        0.9663792861       -0.9802338544
H                -0.7020852529        1.2179947762        1.2040449710
H                -0.4165282401       -0.5117461928        1.3370801718
H                 0.8172686847        1.5027570888       -0.6093689628
H                 1.6462420695        0.5923868651        0.6569536822
H                 3.3043023776       -2.6201463609        0.5867695388
H                 5.1884364349       -2.2294494624        2.1828926586
H                 6.9355606717        1.8690600220        1.3345943339
H                 6.5006579413        3.7188911872       -0.2836687551
H                 4.6754879230        3.4955617787       -1.9624611772
H                 3.2858090464        1.4592822278       -2.0311115211
H                -0.0249027380        0.1119120608       -2.6541777990
H                 0.4180436850       -1.5981993895       -2.6503957798
H                -1.9650316119       -1.0110205928       -2.2095561568
H                -2.5262281076       -2.7871109127       -1.0014727032
H                -0.9355920144       -3.2046673654       -1.6005149546
H                -1.6231712603       -2.4234213693        1.2816137167
H                -1.5474329565       -4.0487252610        0.6430063566
H                 0.8427108114       -2.1713585928        0.9668798423
H                 0.4451920326       -3.5953527902        1.9319008908
H                 0.8236195411       -4.7936103185       -0.0452492429
H                -4.7770236454       -1.0676906985       -0.2621247530
H                -3.7869173175       -0.4563967424       -1.5825416792
H                -5.7630829515        1.0176669787        0.1636102421
H                -5.3371736844        1.2773916059       -1.5297476508


