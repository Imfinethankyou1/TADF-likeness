%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/137479145_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/137479145_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                10.5362344307        1.6790048336       -0.3586614092
C                 9.6952196020        0.4889430867       -0.6826987282
N                10.2132166154       -0.6158043476       -1.1639187729
O                 9.0891022925       -1.4543287001       -1.3183423084
C                 8.0174679356       -0.7466200000       -0.9071266727
C                 6.6992721176       -1.3634252623       -0.9521155947
C                 6.5373850890       -2.6891182086       -1.4203232704
C                 5.2873135981       -3.2915738795       -1.4690218560
C                 4.1748302593       -2.5551958059       -1.0412657022
N                 2.8410477820       -2.9138614104       -0.9877135116
C                 2.2426570028       -1.8409351078       -0.5071777112
N                 0.8811123238       -1.7946321334       -0.2990438481
C                 0.1103410436       -0.7527745330        0.1952441464
C                -1.2825346100       -0.9349571598        0.3200650353
C                -2.0583863153        0.1022182932        0.8179236650
C                -3.5514031128       -0.0841304558        1.0328737506
N                -4.3131217044        1.1053747346        0.6418745467
C                -4.2890704989        1.3168060861       -0.8079237079
C                -5.0816219510        0.2360023081       -1.5777515203
N                -6.2466575946       -0.1418800857       -0.7792836720
C                -6.6718642772       -1.4378903137       -0.7524535260
O                -6.1832621273       -2.3513955373       -1.3979498892
C                -6.7625356752        0.8787346621        0.1142373668
C                -5.6944325257        1.3009894571        1.1893883382
C                -5.9392610844        0.4985606989        2.4763872113
C                -5.8716998422        2.7974335842        1.5085684359
C                -1.4153163543        1.3032678662        1.1661128490
C                -0.0419149619        1.3872446614        1.0028449247
N                 0.7252674578        0.3901008637        0.5302284805
N                 3.0803117401       -0.7944650900       -0.2368516787
C                 4.3565139107       -1.2239204002       -0.5739707043
C                 5.5959947341       -0.6084942337       -0.5189351727
N                 8.3322476048        0.4597626287       -0.5039895062
H                10.4369676022        1.9372304852        0.7010762916
H                10.2081106576        2.5462650294       -0.9415551748
H                11.5859228918        1.4745328766       -0.5806966600
H                 7.4142824617       -3.2383357334       -1.7458383540
H                 5.1653991776       -4.3082935018       -1.8283626785
H                 0.4144413323       -2.6553782011       -0.5514574456
H                -1.7384350356       -1.8785327714        0.0296357682
H                -3.6977126525       -0.2547625518        2.1019364108
H                -3.8714652102       -1.0100990983        0.5254271066
H                -3.2521296895        1.3366604505       -1.1560397123
H                -4.7030372705        2.3088684137       -1.0176840168
H                -4.4956231693       -0.6703601454       -1.7470365519
H                -5.3898658623        0.6123749630       -2.5639112255
H                -7.5306344683       -1.5807855916       -0.0663491941
H                -7.6654737456        0.4959231708        0.6003667858
H                -7.0666655043        1.7521983598       -0.4783657729
H                -5.2001908599        0.7352102189        3.2491101808
H                -6.9235931452        0.7628914421        2.8774269823
H                -5.9310142248       -0.5830598463        2.3041264612
H                -5.6451761837        3.4230779543        0.6386573965
H                -5.2002829696        3.0944371551        2.3201586295
H                -6.9029331781        3.0131883690        1.8150501303
H                -1.9905180253        2.1427080554        1.5394115088
H                 0.4897075645        2.3004292608        1.2624138545
H                 2.7392743061        0.0857537647        0.1363265756
H                 5.7381922958        0.4061303723       -0.1630552831


