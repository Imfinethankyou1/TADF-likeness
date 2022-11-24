%chk=./calculation/no_transfer_pubchem_sampling_5_b3lyp/57319177.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 0.1145206598        0.4177823309       -2.7296254743
C                 0.0720294966        0.5258942254       -1.2052373172
C                 1.4122236200        0.3941273519       -0.5073425306
C                 1.4397450575        0.3439867425        0.8843822840
C                 2.6330408773        0.2677673934        1.5727711597
C                 3.8462990825        0.2378561630        0.8890008225
C                 5.1205756138        0.1539858866        1.6168096766
C                 5.3310363703        0.9079698623        2.7688217393
C                 6.5319565412        0.8325019278        3.4493015595
C                 7.5389717998       -0.0019569446        2.9939847079
C                 7.3387043641       -0.7591188092        1.8518266760
C                 6.1409248927       -0.6802321975        1.1663166068
C                 3.8182135064        0.2898494738       -0.5008436967
C                 2.6212210689        0.3721806933       -1.1884461633
N                -0.9183817205       -0.3897284192       -0.6415644727
C                -2.0323171318        0.0140337333        0.0133590724
O                -2.9227213016       -0.7768113884        0.3001164957
C                -2.1477279772        1.4910024982        0.3995391393
C                -2.4425014325        2.3533478098       -0.8073356732
C                -3.1050172314        1.8462779213       -1.9186527804
C                -3.3926012300        2.6642716532       -2.9974718247
C                -3.0256956028        3.9983726264       -2.9771766987
C                -2.3703741399        4.5126339990       -1.8710511209
C                -2.0810986277        3.6955809511       -0.7937135784
C                -3.2407933566        1.6540965665        1.4341387621
C                -2.9077310027        1.9019052503        2.7582959219
C                -3.8946113652        2.0403873163        3.7181048039
C                -5.2271536488        1.9283565165        3.3627115033
C                -5.5658839517        1.6783678171        2.0440050715
C                -4.5792122530        1.5437619565        1.0843287508
C                -0.8279923068       -1.8320471729       -0.9105075079
C                 0.5792205671       -2.4191658228       -0.8221817577
C                 0.5642906394       -3.9486717044       -0.8536743894
C                -0.0811446129       -4.5928254042        0.3724382025
N                 0.2392451987       -6.0127377176        0.3898921861
C                -1.5013738977       -2.1689118951       -2.2564412842
N                -2.8393407177       -2.0364548105       -2.2062417126
O                -0.9025465448       -2.5165106897       -3.2506700035
H                 0.7187687086        1.2249746316       -3.1358416983
H                 0.5083743869       -0.5337553210       -3.0708633561
H                -0.8951255570        0.5290282952       -3.1204371880
H                -0.2796340490        1.5409266986       -0.9956484830
H                 0.5099683595        0.3424163447        1.4359824428
H                 2.6264351919        0.2094465558        2.6514103482
H                 4.5544665605        1.5726488331        3.1190782897
H                 6.6843136263        1.4280337335        4.3378107274
H                 8.4760121427       -0.0621418773        3.5274916201
H                 8.1192908455       -1.4152402381        1.4950717069
H                 5.9829951997       -1.2852660912        0.2853636675
H                 4.7472530139        0.2847864468       -1.0516939151
H                 2.6418803010        0.4139157789       -2.2654203687
H                -1.2078840550        1.8164353225        0.8582571176
H                -3.3953273471        0.8057069170       -1.9442199326
H                -3.9049467756        2.2567186307       -3.8566581132
H                -3.2502735902        4.6354967171       -3.8193799507
H                -2.0832184968        5.5534590351       -1.8470880325
H                -1.5706824064        4.1007680982        0.0688638798
H                -1.8681157373        1.9865343545        3.0455244422
H                -3.6210991896        2.2336968463        4.7452419864
H                -5.9986617773        2.0342694735        4.1111398419
H                -6.6043911468        1.5873792411        1.7610296840
H                -4.8510213785        1.3507762006        0.0581177342
H                -1.4413733073       -2.2946313012       -0.1282537667
H                 1.1797833840       -2.0718417917       -1.6619378684
H                 1.0516978687       -2.0889530934        0.1033315427
H                 0.0588556784       -4.2898153214       -1.7582027814
H                 1.5963001717       -4.3005266349       -0.8987035713
H                 0.3406364213       -4.1529666174        1.2817976403
H                -1.1677714280       -4.4025279750        0.3764632824
H                -0.1171933479       -6.4574050952       -0.4500393239
H                -0.1956438729       -6.4648800417        1.1868875119
H                -3.2935094800       -1.7748514152       -1.3405531904
H                -3.3871572208       -2.2437430301       -3.0253760807


