%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/148035933.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.7043734453        5.0210449420        2.9905379897
O                 7.4688984168        3.9396157096        2.1001079847
C                 7.6162184390        2.7142900257        2.6070490841
O                 7.9097651919        2.4962108934        3.7534071341
C                 7.3339534379        1.6365116096        1.5799467646
C                 7.8733769443        1.9757235264        0.1842491682
C                 7.3275370758        0.9433906146       -0.8086223552
O                 6.3316041986        1.2088242964       -1.4713631038
N                 7.9281502296       -0.2496695115       -0.9082089886
C                 9.1163628849       -0.7435936482       -0.2363404197
C                 9.1359831345       -2.2324112195       -0.5749870399
C                 8.5325872204       -2.2654461931       -1.9749287878
C                 7.4079970864       -1.2414841069       -1.8625186926
C                 6.1495690059       -1.8551350061       -1.3366347255
N                 6.0160640764       -3.0733875023       -0.8625589880
C                 4.7222819319       -3.1928418376       -0.4924954179
C                 4.0450705784       -2.0140361082       -0.7414931524
C                 2.7144456941       -1.6578944175       -0.5256892948
C                 1.5609286954       -1.3430157048       -0.3432643169
C                 0.2135664603       -1.0023279975       -0.1219386839
C                -0.6684822147       -1.9696349022        0.4180090609
C                -1.9767431798       -1.6605322475        0.6402658476
C                -2.4835565622       -0.3762106702        0.3363730906
C                -3.8323765626       -0.0432177349        0.5426159873
C                -4.3246991285        1.2049188916        0.2199907749
C                -5.7232034269        1.5340955922        0.3644448609
C                -6.4533331279        2.6477975108        0.0069087357
N                -7.7708113965        2.4531697183        0.2496763239
C                -7.8729349341        1.2468025281        0.7492564603
C                -9.1586183028        0.5770421731        1.1336998941
C                -9.7343954565        1.1708933446        2.4286734800
C                -9.0588380478        0.3358928259        3.5128011190
C                -9.0673902802       -1.0592494224        2.8999405301
N                -8.9378014526       -0.8216439193        1.4668281987
C                -8.4246401991       -1.7740085998        0.6667881997
O                -8.0660290489       -2.8559463824        1.1054466691
C                -8.3241030074       -1.4640770495       -0.8295769263
N                -7.1483942006       -2.1284660813       -1.3436226893
C                -6.1002875331       -1.4313620502       -1.8361004582
O                -6.1128039671       -0.2547969558       -2.1040530762
O                -5.0333933922       -2.2493377455       -2.0052894179
C                -3.8870903531       -1.6744511260       -2.6139961011
C                -9.6148451372       -1.9303845460       -1.5424957273
C                -9.6465900058       -1.3698041493       -2.9612600663
C                -9.7268532987       -3.4527320414       -1.5573189182
N                -6.6574359937        0.6584981520        0.8619089042
C                -3.4341761604        2.1685342666       -0.3121679881
C                -2.1201321969        1.8739814074       -0.5166984811
C                -1.6021920429        0.5959629751       -0.2075577781
C                -0.2571455005        0.2592879524       -0.4257599248
N                 4.9817507473       -1.1735382003       -1.2889076143
C                 9.3641201780        2.1652858569        0.0752490693
C                 9.8804855343        2.4922512079       -1.1768983419
C                11.2357027265        2.6931605167       -1.3558657362
C                12.1006897223        2.5790266011       -0.2796528428
C                11.5971397913        2.2634024406        0.9692049847
C                10.2388853559        2.0548557385        1.1475158138
H                 7.1134441335        4.8997050378        3.9004314410
H                 8.7618317081        5.0626300497        3.2592825654
H                 7.4120362650        5.9225229460        2.4586572370
H                 6.2460367563        1.5410777264        1.5101939758
H                 7.7263762960        0.6938728823        1.9631120992
H                 7.4017361821        2.9194972187       -0.1174598005
H                 9.0749609861       -0.5633571783        0.8377173161
H                10.0073480040       -0.2480343924       -0.6386901587
H                10.1400970718       -2.6476225534       -0.5321622295
H                 8.4874292225       -2.7859025617        0.1029762133
H                 8.1443211286       -3.2460041667       -2.2344959434
H                 9.2629830891       -1.9469301242       -2.7190736644
H                 7.1844270308       -0.7305822909       -2.8068997540
H                 4.3451110948       -4.1033433417       -0.0726997023
H                -0.2819434756       -2.9518280424        0.6480925729
H                -2.6501509128       -2.4004432387        1.0500665708
H                -4.4758681176       -0.8129764920        0.9473004017
H                -6.1192888161        3.5705887574       -0.4207031911
H                -9.8685691224        0.6871209289        0.3067415026
H                -9.5177833598        2.2337969283        2.5009300959
H               -10.8141172497        1.0240438912        2.4566627877
H                -8.0354457273        0.6801780673        3.6675755604
H                -9.5864748142        0.3714748820        4.4628004749
H               -10.0144195270       -1.5737577562        3.0950367846
H                -8.2547807626       -1.7027597691        3.2430976606
H                -8.1834644906       -0.3969023555       -1.0184009606
H                -6.9883687733       -3.0682532419       -1.0087157564
H                -3.8761831479       -1.9206977212       -3.6778133964
H                -3.9026998551       -0.5898617767       -2.4907805274
H                -3.0128340198       -2.0988046254       -2.1232563536
H               -10.4599849440       -1.5194913027       -0.9772856898
H               -10.5413770854       -1.7052296709       -3.4800781555
H                -8.7750143753       -1.7093939298       -3.5151049646
H                -9.6382030096       -0.2824898866       -2.9480121421
H                -8.9811708836       -3.8791199614       -2.2235990367
H               -10.7111196453       -3.7506668975       -1.9108105049
H                -9.5787890576       -3.8617671562       -0.5610464324
H                -6.4711413520       -0.2745964247        1.1893404907
H                -3.8179132913        3.1447650308       -0.5616376860
H                -1.4510432161        2.6165352060       -0.9276397764
H                 0.4133519917        0.9998809668       -0.8382829768
H                 4.8760116011       -0.2013846941       -1.5496050027
H                 9.2047492141        2.5877588419       -2.0152744455
H                11.6178589930        2.9434904306       -2.3345546182
H                13.1599307893        2.7388701876       -0.4143946840
H                12.2632459957        2.1772897088        1.8149581268
H                 9.8766493859        1.8072666236        2.1328236016


