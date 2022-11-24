%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/68470225_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/68470225_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.5328769936        1.2879912977        1.3310118356
O                 6.2947961021        1.2851157399        2.0173382219
C                 5.2055415820        0.8108159387        1.3509507164
C                 5.2077855812        0.3148358152        0.0565006233
C                 3.9726478664       -0.1219231191       -0.4518930686
C                 2.8122854210       -0.0439102565        0.3390565841
N                 2.8192683736        0.4228666319        1.6048560857
C                 3.9679092372        0.8400029857        2.0856282607
O                 4.0466946293        1.3363928632        3.3389139032
C                 2.8183741278        1.3954962731        4.0729679279
C                 1.7190991960       -0.5529532405       -0.4479744457
C                 0.3436895196       -0.7953553194       -0.0335423790
C                -0.3308635930       -1.9902413903       -0.1130900873
C                -1.6055914870       -1.8393176117        0.5163210952
C                -2.7054082517       -2.6721287062        0.7874158215
C                -2.6822304467       -4.1424008642        0.3958377530
O                -2.2653503836       -4.3379767165       -0.9463062866
C                -3.4678304766       -4.3168844295       -1.7137098692
C                -4.4483827665       -5.0465008161       -0.7917459467
O                -3.9708921756       -4.7337624139        0.5207024448
C                -3.7766304523       -2.1028227758        1.4739789435
C                -3.7300766225       -0.7555873216        1.8623152093
N                -2.6927130498        0.0587640494        1.6315640974
C                -1.6795072364       -0.5016811069        0.9845374965
N                -0.4842784258        0.1429781064        0.6364052326
S                -0.2968519858        1.8793329704        0.7050052922
O                -0.4163358985        2.2757891009        2.1031128599
O                 0.8870153083        2.1859793224       -0.0897340146
C                -1.7167635305        2.4968543825       -0.1976129373
C                -1.6398801925        2.5779751545       -1.5895228063
C                -2.7184403967        3.1054612093       -2.2948458578
C                -3.8640353188        3.5659527590       -1.6292458994
C                -5.0124249167        4.1724235316       -2.3998820890
C                -3.9062199473        3.4781076882       -0.2307777964
C                -2.8419120064        2.9455181003        0.4937088852
C                 2.2675269627       -0.9244685038       -1.6585183399
N                 3.6241554565       -0.6687490184       -1.6736239471
C                 4.5164769390       -0.8707023074       -2.7950103454
H                 7.4926460174        1.9128703850        0.4277474220
H                 7.8420984172        0.2712693446        1.0499314972
H                 8.2615646702        1.7066542692        2.0274276678
H                 6.1163052504        0.2712529847       -0.5327816075
H                 2.0657443707        1.9848054721        3.5419476985
H                 2.4158182719        0.3912412777        4.2409552273
H                 3.0787431509        1.8639317057        5.0239259653
H                 0.0871911142       -2.8983793413       -0.5220400705
H                -1.9723792763       -4.6918527509        1.0272816925
H                -3.2764929477       -4.8284660308       -2.6595796867
H                -3.7897312432       -3.2847066133       -1.9120487300
H                -4.4163905421       -6.1331093629       -0.9444822322
H                -5.4816517271       -4.6994056841       -0.8936055471
H                -4.6442514661       -2.7082633840        1.7098863037
H                -4.5711908830       -0.3138366015        2.3925248515
H                -0.7451328898        2.2455724872       -2.1050820693
H                -2.6667162075        3.1691450099       -3.3790596920
H                -5.9669951235        4.0146048876       -1.8873930479
H                -4.8795254706        5.2565065754       -2.5142356906
H                -5.0902809579        3.7468018975       -3.4057242659
H                -4.7868567382        3.8303564022        0.3007674095
H                -2.8759682706        2.8658981701        1.5730215868
H                 1.7822987675       -1.3447112622       -2.5286662245
H                 4.9026114050        0.0835778233       -3.1744406139
H                 5.3648037420       -1.5036530199       -2.5087262638
H                 3.9702108306       -1.3693946444       -3.5986477537


