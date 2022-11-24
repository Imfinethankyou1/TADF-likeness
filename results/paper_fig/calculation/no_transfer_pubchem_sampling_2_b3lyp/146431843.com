%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/146431843.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -6.6625140658        4.0332319341        2.2730537549
O                -5.8086034121        2.9236816181        2.0262323465
C                -5.4421210524        2.7370265727        0.7526017826
O                -5.8006808331        3.4574221010       -0.1432294600
C                -4.4944707278        1.5506941775        0.5436924572
C                -5.2486645775        0.3166492298        0.0776242847
C                -6.6298012529        0.2475938317        0.2006204860
C                -7.3314812403       -0.8676312677       -0.2201691529
C                -6.6472351823       -1.9359115098       -0.7692646380
C                -5.2708019712       -1.8795915762       -0.8829315020
C                -4.5489816023       -0.7686589595       -0.4616446736
C                -3.0477415813       -0.7678697230       -0.5921829667
C                -2.3902318803       -1.9757627440        0.0837808982
C                -0.8770010047       -1.7227683485        0.1216278302
O                -0.4359327269       -0.8984530713       -0.9455788346
C                -1.1205964206       -1.1596588259       -2.1475968313
C                -2.6003853303       -0.7397138263       -2.0633400247
N                -3.6288123901        1.1952153329        1.6399108080
C                -4.1833681698        0.5713978081        2.8235502918
C                -2.9757992685        0.4569932986        3.7628904229
C                -1.9116520896        1.4152346718        3.1889882315
O                -0.8602239253        0.6115204964        2.7074969210
C                 0.3839045217        1.2331119792        2.4538306988
C                 0.5309688649        1.7950441578        1.0343383204
C                 1.9857715656        1.8597895136        0.5699990658
C                 2.6074924157        0.4723407208        0.3583638032
C                 3.8458985103        0.5568041153       -0.4994067446
C                 3.8238968012        0.6282710081       -1.8331087548
C                 2.5804859089        0.5579621322       -2.6672364634
C                 5.0895498072        0.8488479125       -2.6150877906
C                 5.6749354324       -0.4639726950       -3.1236592157
C                 6.2037526710       -1.4301762375       -2.0933161508
C                 6.4589763736       -1.2238742875       -0.7889645084
C                 6.2114513931       -0.0570377572        0.0513548901
C                 5.0982471750        0.6567553267        0.2428584645
C                 7.1487672722       -2.3128762297        0.0219449882
C                 6.9623845247       -3.6903640044       -0.5964015206
C                 7.3442177358       -3.6151487207       -2.0661059034
N                 6.4739797454       -2.6563179628       -2.7004559433
C                -2.6802141326        2.1793638671        2.0970808045
H                -7.6055634434        3.9114584665        1.7364659317
H                -6.8326874439        4.0539956486        3.3463826790
H                -6.1877041225        4.9576023700        1.9384635778
H                -3.8490795542        1.8806331658       -0.2849764041
H                -7.1728619619        1.0768751022        0.6285199570
H                -8.4059279651       -0.8993615845       -0.1188376491
H                -7.1827254025       -2.8119092358       -1.1045664472
H                -4.7455867222       -2.7220439897       -1.3080144588
H                -2.6416362045        0.1080818050       -0.0878206704
H                -2.6129121149       -2.8898214492       -0.4695752250
H                -2.7635452762       -2.0912062809        1.1003114839
H                -0.6085350843       -1.1831213698        1.0323372275
H                -0.3291670066       -2.6777692208        0.0938608542
H                -1.0387154491       -2.2288175195       -2.3935776276
H                -0.5962740615       -0.5917437960       -2.9193938155
H                -3.2077409914       -1.4203534552       -2.6617446151
H                -2.7346736857        0.2663544233       -2.4629862083
H                -4.6062903312       -0.4002621057        2.5624322662
H                -4.9701651361        1.1826953356        3.2882251811
H                -3.2372725181        0.7130858478        4.7872302256
H                -2.5559328329       -0.5469058257        3.7472321452
H                -1.5194533846        2.1097451664        3.9476256111
H                 0.5743511062        2.0166488859        3.2015835584
H                 1.1163189191        0.4343664343        2.5993185187
H                -0.0217056064        1.1477520393        0.3513803266
H                 0.1027574422        2.7994467533        0.9866083417
H                 2.0145164688        2.3972282088       -0.3796919505
H                 2.5807053295        2.4285698017        1.2887953345
H                 2.8771829044        0.0320016296        1.3198580027
H                 1.8659978060       -0.1754222999       -0.1090276612
H                 1.7384295939        0.1514642082       -2.1169473703
H                 2.3092665767        1.5542178233       -3.0215267446
H                 2.7575912549       -0.0637931866       -3.5451263442
H                 5.8329298359        1.3614574325       -2.0044845798
H                 4.8656203676        1.4761028022       -3.4831537189
H                 6.5006689524       -0.2404273804       -3.8133887681
H                 4.9060673586       -0.9856685311       -3.7031615480
H                 7.0318488126        0.1278825344        0.7412251821
H                 5.0860850838        1.3719593783        1.0585461684
H                 6.7437618872       -2.3128082523        1.0368244455
H                 8.2180051300       -2.0837047321        0.0992961731
H                 5.9147237272       -3.9848709659       -0.5216802887
H                 7.5728541607       -4.4329386693       -0.0801662771
H                 7.2048884299       -4.5828850746       -2.5584607375
H                 8.4065859035       -3.3345225888       -2.1585577229
H                 6.5783326133       -2.6037133784       -3.7032753459
H                -2.0410610784        2.4965688326        1.2743595348
H                -3.1596371677        3.0687137371        2.5437529803


