%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/101022871.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -6.7830719378       -1.8711702948        1.7586344863
C                -5.3881590918       -2.0492768416        2.3538185143
C                -5.1124855949       -3.5208848151        2.7063215323
C                -4.3886747726       -4.2331128440        1.5696307768
C                -2.9627075469       -3.6887187325        1.3854290102
C                -2.3857375579       -4.1516936872        0.0286248014
C                -0.8561191199       -4.1506301972        0.0274081870
C                -2.9122188944       -3.3772519189       -1.1775734191
C                -2.9710936878       -2.1668117670        1.6663853208
O                -1.9639284795       -1.4738274409        0.9203766318
C                -1.0501857240       -0.7729747989        1.5847024022
O                -0.9741552240       -0.6990492615        2.7836707089
C                -0.1277128633       -0.0675229048        0.6068028871
C                -0.9133768999        1.1033658409       -0.0540074276
C                -1.1886339400        2.1049087259        1.0294214419
C                -2.4257197667        2.5400874531        1.4580391489
C                -2.5017376501        3.4633416101        2.4865185795
C                -1.3616719198        3.9675359204        3.1070962122
C                -1.5131180702        4.9753837020        4.2365407404
C                -2.3249268160        4.3350120991        5.3712824146
C                -0.1669850623        5.4258024181        4.8100780880
C                -2.2454308240        6.2165655211        3.7071368593
C                -0.1213879209        3.5174656654        2.6643213447
C                -0.0388854412        2.5990646765        1.6368932234
C                 1.2271278830        2.0105226717        1.0813360511
C                 1.2014666282        2.2225969669       -0.4050402194
C                 2.1768110506        2.8178245970       -1.1757434438
C                 1.9821525179        2.9557943273       -2.5435013464
C                 0.8225896020        2.5090372742       -3.1650081452
C                 0.5480309249        2.7136906060       -4.6478909871
C                 0.1751370190        1.3794594631       -5.3085325497
C                 1.7504211005        3.2928412398       -5.3985552003
C                -0.6231423058        3.6983597496       -4.7823437403
C                -0.1458549845        1.8895179314       -2.3739525209
C                 0.0430448091        1.7476847141       -1.0165514164
C                 1.1164530275        0.4789903566        1.3133166021
C                 2.4219813426       -0.1829146647        0.9145533001
O                 3.5011043946        0.2706957783        1.2020172736
O                 2.2394661422       -1.3191898554        0.2480659006
C                 3.3963432350       -2.0501889459       -0.1874101783
C                 3.2378879827       -3.4854349874        0.3109021858
C                 4.3908039703       -4.3868212421       -0.1367549404
C                 5.6883135808       -4.0660484839        0.6028750291
C                 4.5509156118       -4.2929500974       -1.6566458805
C                 4.7003963041       -2.8517081689       -2.1359178709
C                 3.4946460540       -2.0080443899       -1.7181351371
C                 3.5123976216       -0.5668762425       -2.2683538899
C                 4.7560545957        0.2309207830       -1.8833419637
C                 3.3459910966       -0.5592710356       -3.7872449660
C                -4.3284957909       -1.5307229735        1.3728461027
H                -6.9074038368       -2.5002083519        0.8797491305
H                -6.9421413242       -0.8359640418        1.4649000140
H                -7.5465068888       -2.1415720017        2.4845143917
H                -5.3314665787       -1.4570761993        3.2728791407
H                -4.4978041738       -3.5814903924        3.6064006821
H                -6.0552958673       -4.0272714352        2.9197851622
H                -4.3331917368       -5.3032314808        1.7780909863
H                -4.9606769697       -4.1104857635        0.6496695138
H                -2.3317120513       -4.1470241639        2.1558814750
H                -2.7050340334       -5.1967217972       -0.0822537772
H                -0.4872857269       -4.6038314078       -0.8911784334
H                -0.4706803016       -4.7243851023        0.8680644283
H                -0.4631576815       -3.1408240019        0.0916902086
H                -2.6091820889       -2.3364074695       -1.1267465397
H                -2.5084732597       -3.8090855768       -2.0914651849
H                -3.9966486817       -3.4294623411       -1.2382309767
H                -2.7430060582       -2.0054677596        2.7278817650
H                 0.1410682822       -0.7772137704       -0.1789217492
H                -1.8206491953        0.7338986509       -0.5354582322
H                -3.3287193674        2.1650106244        0.9961294439
H                -3.4789222135        3.7903938706        2.8083365067
H                -2.4181150853        5.0291314115        6.2040546153
H                -3.3238051707        4.0644850123        5.0406627372
H                -1.8280706192        3.4341183658        5.7247340002
H                 0.4409794201        5.9090514061        4.0487391244
H                -0.3362912173        6.1439013689        5.6101336948
H                 0.3819202045        4.5841125817        5.2254956428
H                -3.2385580277        5.9669470719        3.3442430682
H                -2.3480188445        6.9569066432        4.4978827354
H                -1.6853114520        6.6616036385        2.8872717206
H                 0.7929615619        3.8703661067        3.1126234847
H                 2.1308583557        2.4009972271        1.5492112564
H                 3.0867639560        3.1770273408       -0.7182482031
H                 2.7593304328        3.4331311672       -3.1178596843
H                 0.0042739648        1.5273109450       -6.3729553081
H                -0.7295330007        0.9590202187       -4.8792466093
H                 0.9793710707        0.6584283732       -5.1859842749
H                 1.5069447164        3.3880387270       -6.4549278696
H                 2.0081182838        4.2812071574       -5.0258147830
H                 2.6178930640        2.6422180794       -5.3114421044
H                -0.8438310771        3.8800218456       -5.8322186528
H                -0.3731682990        4.6469991247       -4.3117668031
H                -1.5186084687        3.3090625942       -4.3052274796
H                -1.0620925637        1.5236099694       -2.8131819527
H                 1.0037842482        0.3109193963        2.3930935963
H                 4.2760228162       -1.5815882438        0.2638582444
H                 3.1676602452       -3.4774451136        1.4006122600
H                 2.2949975215       -3.8748801136       -0.0800582472
H                 4.1201726090       -5.4199488331        0.1126430704
H                 5.5470615465       -4.1809802531        1.6750964361
H                 6.4768967314       -4.7460920059        0.2879994323
H                 6.0183873135       -3.0495360762        0.4127707994
H                 5.4210521086       -4.8744114380       -1.9684049987
H                 3.6705679518       -4.7337445675       -2.1311460024
H                 5.6119250776       -2.4221423507       -1.7217931285
H                 4.7910983255       -2.8486068587       -3.2225691068
H                 2.5873805665       -2.4867689929       -2.1085022995
H                 2.6415454964       -0.0509844110       -1.8456072003
H                 4.8542707302        0.3120014931       -0.8046097756
H                 5.6563678637       -0.2216022429       -2.2917649376
H                 4.6717282202        1.2386568572       -2.2838988029
H                 3.1935602674        0.4613281325       -4.1299891508
H                 2.4867183146       -1.1558311948       -4.0869988806
H                 4.2311201214       -0.9525812137       -4.2814731495
H                -4.6191635862       -1.7516944931        0.3465811508
H                -4.2341905479       -0.4467174411        1.4657228661


