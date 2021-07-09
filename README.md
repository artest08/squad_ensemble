# squad_ensemble

###tkwn distil
n = 20 and thres = 0, max_length = 30
{'exact': 62.68845279204919, 'f1': 67.12506411683287, 'total': 11873, 'HasAns_exact': 62.68845279204919, 'HasAns_f1': 67.12506411683287, 'HasAns_total': 11873, 'best_exact': 62.68845279204919, 'best_exact_thresh': 0.0, 'best_f1': 67.12506411683287, 'best_f1_thresh': 0.0}

n = 20 and thres = 0, max_length = 30, geri bestleri siralamanin pek bir mantigi yok
{'exact': 62.68845279204919, 'f1': 67.12506411683287, 'total': 11873, 'HasAns_exact': 62.68845279204919, 'HasAns_f1': 67.12506411683287, 'HasAns_total': 11873, 'best_exact': 62.68845279204919, 'best_exact_thresh': 0.0, 'best_f1': 67.12506411683287, 'best_f1_thresh': 0.0}

###distilbert-base-uncased-distilled-squad
{'exact': 35.18066200623263, 'f1': 40.43015859062201, 'total': 11873, 'HasAns_exact': 35.18066200623263, 'HasAns_f1': 40.43015859062201, 'HasAns_total': 11873, 'best_exact': 35.18066200623263, 'best_exact_thresh': 0.0, 'best_f1': 40.43015859062201, 'best_f1_thresh': 0.0}

###distilbert-base-uncased
{'exact': 48.058620399225134, 'f1': 48.58858209125072, 'total': 11873, 'HasAns_exact': 48.058620399225134, 'HasAns_f1': 48.58858209125072, 'HasAns_total': 11873, 'best_exact': 48.058620399225134, 'best_exact_thresh': 0.0, 'best_f1': 48.58858209125072, 'best_f1_thresh': 0.0}

###bert-base-uncased
{'exact': 68.48311294533816, 'f1': 73.00700540875151, 'total': 11873, 'HasAns_exact': 68.48311294533816, 'HasAns_f1': 73.00700540875151, 'HasAns_total': 11873, 'best_exact': 68.48311294533816, 'best_exact_thresh': 0.0, 'best_f1': 73.00700540875151, 'best_f1_thresh': 0.0}

### Roberta distil
{'exact': 54.13122210056431, 'f1': 58.22665549068309, 'total': 11873, 'HasAns_exact': 54.13122210056431, 'HasAns_f1': 58.22665549068309, 'HasAns_total': 11873, 'best_exact': 54.13122210056431, 'best_exact_thresh': 0.0, 'best_f1': 58.22665549068309, 'best_f1_thresh': 0.0}


after correction
the best validation loss is loaded: 1.5705734861407368
{'exact': 53.651141244841234, 'f1': 58.23631159198432, 'total': 11873, 'HasAns_exact': 53.651141244841234, 'HasAns_f1': 58.23631159198432, 'HasAns_total': 11873, 'best_exact': 53.651141244841234, 'best_exact_thresh': 0.0, 'best_f1': 58.23631159198432, 'best_f1_thresh': 0.0}

after correction 3 epochs
the best validation loss is loaded: 1.699180577514806
{'exact': 57.07908700412701, 'f1': 61.514399272866115, 'total': 11873, 'HasAns_exact': 57.07908700412701, 'HasAns_f1': 61.514399272866115, 'HasAns_total': 11873, 'best_exact': 57.07908700412701, 'best_exact_thresh': 0.0, 'best_f1': 61.514399272866115, 'best_f1_thresh': 0.0}


###ktrapeznikov/albert-xlarge-v2-squad-v2
{'exact': 83.62671607849744, 'f1': 87.04753946171401, 'total': 11873, 'HasAns_exact': 83.62671607849744, 'HasAns_f1': 87.04753946171401, 'HasAns_total': 11873, 'best_exact': 83.62671607849744, 'best_exact_thresh': 0.0, 'best_f1': 87.04753946171401, 'best_f1_thresh': 0.0}

###twmkn9/albert-base-v2-squad2
{'exact': 77.14141329065949, 'f1': 81.02153932982586, 'total': 11873, 'HasAns_exact': 77.14141329065949, 'HasAns_f1': 81.02153932982586, 'HasAns_total': 11873, 'best_exact': 77.14141329065949, 'best_exact_thresh': 0.0, 'best_f1': 81.02153932982586, 'best_f1_thresh': 0.0}

### average of two model above
{'exact': 82.599174597827, 'f1': 85.9505844079406, 'total': 11873, 'HasAns_exact': 82.599174597827, 'HasAns_f1': 85.9505844079406, 'HasAns_total': 11873, 'best_exact': 82.599174597827, 'best_exact_thresh': 0.0, 'best_f1': 85.9505844079406, 'best_f1_thresh': 0.0}

### max of two models
{'exact': 77.2845953002611, 'f1': 81.16394091512976, 'total': 11873, 'HasAns_exact': 77.2845953002611, 'HasAns_f1': 81.16394091512976, 'HasAns_total': 11873, 'best_exact': 77.2845953002611, 'best_exact_thresh': 0.0, 'best_f1': 81.16394091512976, 'best_f1_thresh': 0.0}


### burda distil ile albert predleri ayni sepete kondu
{'exact': 66.7059715320475, 'f1': 70.97688326771683, 'total': 11873, 'HasAns_exact': 66.7059715320475, 'HasAns_f1': 70.97688326771683, 'HasAns_total': 11873, 'best_exact': 66.7059715320475, 'best_exact_thresh': 0.0, 'best_f1': 70.97688326771683, 'best_f1_thresh': 0.0}

###Burda robertadistil ile albert ayni sepete kondu
{'exact': 76.69502231954856, 'f1': 80.49589542965701, 'total': 11873, 'HasAns_exact': 76.69502231954856, 'HasAns_f1': 80.49589542965701, 'HasAns_total': 11873, 'best_exact': 76.69502231954856, 'best_exact_thresh': 0.0, 'best_f1': 80.49589542965701, 'best_f1_thresh': 0.0}

###Burda robertadistil ile bertdistil ayni sepete kondu default params
{'exact': 63.54754484965889, 'f1': 67.7574054718841, 'total': 11873, 'HasAns_exact': 63.54754484965889, 'HasAns_f1': 67.7574054718841, 'HasAns_total': 11873, 'best_exact': 63.54754484965889, 'best_exact_thresh': 0.0, 'best_f1': 67.7574054718841, 'best_f1_thresh': 0.0}

###Burda robertadistil ile bertdistil ayni sepete kondu default params
best_count = 3
threshold = -4
max_answer_length = 20
{'exact': 66.77335130127179, 'f1': 70.19062418341285, 'total': 11873, 'HasAns_exact': 66.77335130127179, 'HasAns_f1': 70.19062418341285, 'HasAns_total': 11873, 'best_exact': 66.77335130127179, 'best_exact_thresh': 0.0, 'best_f1': 70.19062418341285, 'best_f1_thresh': 0.0}

###RobeertaDistil yukardaki parameter
{'exact': 60.1869788595974, 'f1': 63.636230956540416, 'total': 11873, 'HasAns_exact': 60.1869788595974, 'HasAns_f1': 63.636230956540416, 'HasAns_total': 11873, 'best_exact': 60.1869788595974, 'best_exact_thresh': 0.0, 'best_f1': 63.636230956540416, 'best_f1_thresh': 0.0}

### BERT Distil yukardaki parametre
{'exact': 65.05516718605239, 'f1': 68.63735322541908, 'total': 11873, 'HasAns_exact': 65.05516718605239, 'HasAns_f1': 68.63735322541908, 'HasAns_total': 11873, 'best_exact': 65.05516718605239, 'best_exact_thresh': 0.0, 'best_f1': 68.63735322541908, 'best_f1_thresh': 0.0}


###Albert base v2 kendiminki 
{'exact': 57.35702855217721, 'f1': 61.93953776090138, 'total': 11873, 'HasAns_exact': 57.35702855217721, 'HasAns_f1': 61.93953776090138, 'HasAns_total': 11873, 'best_exact': 57.35702855217721, 'best_exact_thresh': 0.0, 'best_f1': 61.93953776090138, 'best_f1_thresh': 0.0}

###Albert two models
{'exact': 77.25090541564896, 'f1': 81.13948373354474, 'total': 11873, 'HasAns_exact': 77.25090541564896, 'HasAns_f1': 81.13948373354474, 'HasAns_total': 11873, 'best_exact': 77.25090541564896, 'best_exact_thresh': 0.0, 'best_f1': 81.13948373354474, 'best_f1_thresh': 0.0}

###Distilbert metric trialswith respect to epochs
best_count = 20
threshold = 0
max_answer_length = 30
epoch:0 
{'exact': 53.659563715994274, 'f1': 58.08140170228553, 'total': 11873, 'HasAns_exact': 53.659563715994274, 'HasAns_f1': 58.08140170228553, 'HasAns_total': 11873, 'best_exact': 53.659563715994274, 'best_exact_thresh': 0.0, 'best_f1': 58.08140170228553, 'best_f1_thresh': 0.0}

epoch:1
{'exact': 54.5944580139813, 'f1': 59.07648040754826, 'total': 11873, 'HasAns_exact': 54.5944580139813, 'HasAns_f1': 59.07648040754826, 'HasAns_total': 11873, 'best_exact': 54.5944580139813, 'best_exact_thresh': 0.0, 'best_f1': 59.07648040754826, 'best_f1_thresh': 0.0}

epoch:2
{'exact': 54.19017939863556, 'f1': 58.858086562808204, 'total': 11873, 'HasAns_exact': 54.19017939863556, 'HasAns_f1': 58.858086562808204, 'HasAns_total': 11873, 'best_exact': 54.19017939863556, 'best_exact_thresh': 0.0, 'best_f1': 58.858086562808204, 'best_f1_thresh': 0.0}

epoch:3
{'exact': 57.272803840646844, 'f1': 61.70116155740647, 'total': 11873, 'HasAns_exact': 57.272803840646844, 'HasAns_f1': 61.70116155740647, 'HasAns_total': 11873, 'best_exact': 57.272803840646844, 'best_exact_thresh': 0.0, 'best_f1': 61.70116155740647, 'best_f1_thresh': 0.0}

epoch: 4
{'exact': 57.76130716752295, 'f1': 62.47175338357671, 'total': 11873, 'HasAns_exact': 57.76130716752295, 'HasAns_f1': 62.47175338357671, 'HasAns_total': 11873, 'best_exact': 57.76130716752295, 'best_exact_thresh': 0.0, 'best_f1': 62.47175338357671, 'best_f1_thresh': 0.0}

epoch: 5
{'exact': 55.3609028889076, 'f1': 60.370452794616895, 'total': 11873, 'HasAns_exact': 55.3609028889076, 'HasAns_f1': 60.370452794616895, 'HasAns_total': 11873, 'best_exact': 55.3609028889076, 'best_exact_thresh': 0.0, 'best_f1': 60.370452794616895, 'best_f1_thresh': 0.0}

epoch: 6
{'exact': 49.970521350964376, 'f1': 55.70500667801027, 'total': 11873, 'HasAns_exact': 49.970521350964376, 'HasAns_f1': 55.70500667801027, 'HasAns_total': 11873, 'best_exact': 49.970521350964376, 'best_exact_thresh': 0.0, 'best_f1': 55.70500667801027, 'best_f1_thresh': 0.0}


epoch: 0
{'exact': 59.03310031163143, 'f1': 62.31884667540191, 'total': 11873, 'HasAns_exact': 59.03310031163143, 'HasAns_f1': 62.31884667540191, 'HasAns_total': 11873, 'best_exact': 59.03310031163143, 'best_exact_thresh': 0.0, 'best_f1': 62.31884667540191, 'best_f1_thresh': 0.0}

epoch: 2
the best validation loss is loaded: 1.5794438317544521
{'exact': 62.01465509980628, 'f1': 65.40594064983395, 'total': 11873, 'HasAns_exact': 62.01465509980628, 'HasAns_f1': 65.40594064983395, 'HasAns_total': 11873, 'best_exact': 62.01465509980628, 'best_exact_thresh': 0.0, 'best_f1': 65.40594064983395, 'best_f1_thresh': 0.0}


epoch: 3
the best validation loss is loaded: 1.5392952132389968
{'exact': 58.915185715488924, 'f1': 63.49344584059472, 'total': 11873, 'HasAns_exact': 58.915185715488924, 'HasAns_f1': 63.49344584059472, 'HasAns_total': 11873, 'best_exact': 58.915185715488924, 'best_exact_thresh': 0.0, 'best_f1': 63.49344584059472, 'best_f1_thresh': 0.0}

epoch: 4
the best validation loss is loaded: 1.746456673460756
{'exact': 55.88309610039585, 'f1': 61.262920152463906, 'total': 11873, 'HasAns_exact': 55.88309610039585, 'HasAns_f1': 61.262920152463906, 'HasAns_total': 11873, 'best_exact': 55.88309610039585, 'best_exact_thresh': 0.0, 'best_f1': 61.262920152463906, 'best_f1_thresh': 0.0}
---------------------------------


### Distil

thres 0
{'exact': 62.68845279204919, 'f1': 67.12506411683287, 'total': 11873, 'HasAns_exact': 62.68845279204919, 'HasAns_f1': 67.12506411683287, 'HasAns_total': 11873, 'best_exact': 62.68845279204919, 'best_exact_thresh': 0.0, 'best_f1': 67.12506411683287, 'best_f1_thresh': 0.0}

threshold = -1
{'exact': 63.581234734271035, 'f1': 67.85004046006962, 'total': 11873, 'HasAns_exact': 63.581234734271035, 'HasAns_f1': 67.85004046006962, 'HasAns_total': 11873, 'best_exact': 63.581234734271035, 'best_exact_thresh': 0.0, 'best_f1': 67.85004046006962, 'best_f1_thresh': 0.0}

threshold = -2
{'exact': 64.22134254190179, 'f1': 68.29317252764919, 'total': 11873, 'HasAns_exact': 64.22134254190179, 'HasAns_f1': 68.29317252764919, 'HasAns_total': 11873, 'best_exact': 64.22134254190179, 'best_exact_thresh': 0.0, 'best_f1': 68.29317252764919, 'best_f1_thresh': 0.0}

threshold = -3
{'exact': 64.62562115724754, 'f1': 68.50948662532168, 'total': 11873, 'HasAns_exact': 64.62562115724754, 'HasAns_f1': 68.50948662532168, 'HasAns_total': 11873, 'best_exact': 64.62562115724754, 'best_exact_thresh': 0.0, 'best_f1': 68.50948662532168, 'best_f1_thresh': 0.0}

threshold = -4
{'exact': 65.00463235913416, 'f1': 68.62166995311908, 'total': 11873, 'HasAns_exact': 65.00463235913416, 'HasAns_f1': 68.62166995311908, 'HasAns_total': 11873, 'best_exact': 65.00463235913416, 'best_exact_thresh': 0.0, 'best_f1': 68.62166995311908, 'best_f1_thresh': 0.0}

threshold = -5
{'exact': 65.08885707066453, 'f1': 68.4585443407459, 'total': 11873, 'HasAns_exact': 65.08885707066453, 'HasAns_f1': 68.4585443407459, 'HasAns_total': 11873, 'best_exact': 65.08885707066453, 'best_exact_thresh': 0.0, 'best_f1': 68.4585443407459, 'best_f1_thresh': 0.0}

threshold = -6
{'exact': 64.81933799376738, 'f1': 67.93764806190592, 'total': 11873, 'HasAns_exact': 64.81933799376738, 'HasAns_f1': 67.93764806190592, 'HasAns_total': 11873, 'best_exact': 64.81933799376738, 'best_exact_thresh': 0.0, 'best_f1': 67.93764806190592, 'best_f1_thresh': 0.0}

----------------------------------------
best_count = 1
{'exact': 65.02989977259328, 'f1': 68.56549816196035, 'total': 11873, 'HasAns_exact': 65.02989977259328, 'HasAns_f1': 68.56549816196035, 'HasAns_total': 11873, 'best_exact': 65.02989977259328, 'best_exact_thresh': 0.0, 'best_f1': 68.56549816196035, 'best_f1_thresh': 0.0}

best_count = 3
{'exact': 65.0130548302872, 'f1': 68.63811382537025, 'total': 11873, 'HasAns_exact': 65.0130548302872, 'HasAns_f1': 68.63811382537025, 'HasAns_total': 11873, 'best_exact': 65.0130548302872, 'best_exact_thresh': 0.0, 'best_f1': 68.63811382537025, 'best_f1_thresh': 0.0}

best_count = 5
{'exact': 65.00463235913416, 'f1': 68.62166995311908, 'total': 11873, 'HasAns_exact': 65.00463235913416, 'HasAns_f1': 68.62166995311908, 'HasAns_total': 11873, 'best_exact': 65.00463235913416, 'best_exact_thresh': 0.0, 'best_f1': 68.62166995311908, 'best_f1_thresh': 0.0}

best_count = 10
{'exact': 65.00463235913416, 'f1': 68.62166995311908, 'total': 11873, 'HasAns_exact': 65.00463235913416, 'HasAns_f1': 68.62166995311908, 'HasAns_total': 11873, 'best_exact': 65.00463235913416, 'best_exact_thresh': 0.0, 'best_f1': 68.62166995311908, 'best_f1_thresh': 0.0}

best_count = 20
{'exact': 65.00463235913416, 'f1': 68.62166995311908, 'total': 11873, 'HasAns_exact': 65.00463235913416, 'HasAns_f1': 68.62166995311908, 'HasAns_total': 11873, 'best_exact': 65.00463235913416, 'best_exact_thresh': 0.0, 'best_f1': 68.62166995311908, 'best_f1_thresh': 0.0}
-----------------------------

max_answer_length = 5
{'exact': 63.72441674387265, 'f1': 66.90497693616494, 'total': 11873, 'HasAns_exact': 63.72441674387265, 'HasAns_f1': 66.90497693616494, 'HasAns_total': 11873, 'best_exact': 63.72441674387265, 'best_exact_thresh': 0.0, 'best_f1': 66.90497693616494, 'best_f1_thresh': 0.0}

max_answer_length = 10
{'exact': 65.04674471489935, 'f1': 68.47732544853113, 'total': 11873, 'HasAns_exact': 65.04674471489935, 'HasAns_f1': 68.47732544853113, 'HasAns_total': 11873, 'best_exact': 65.04674471489935, 'best_exact_thresh': 0.0, 'best_f1': 68.47732544853113, 'best_f1_thresh': 0.0}

max_answer_length = 20
{'exact': 65.05516718605239, 'f1': 68.63735322541908, 'total': 11873, 'HasAns_exact': 65.05516718605239, 'HasAns_f1': 68.63735322541908, 'HasAns_total': 11873, 'best_exact': 65.05516718605239, 'best_exact_thresh': 0.0, 'best_f1': 68.63735322541908, 'best_f1_thresh': 0.0}

max_answer_length = 30
{'exact': 65.0130548302872, 'f1': 68.63811382537025, 'total': 11873, 'HasAns_exact': 65.0130548302872, 'HasAns_f1': 68.63811382537025, 'HasAns_total': 11873, 'best_exact': 65.0130548302872, 'best_exact_thresh': 0.0, 'best_f1': 68.63811382537025, 'best_f1_thresh': 0.0}

max_answer_length = 50
{'exact': 64.962520003369, 'f1': 68.62438092233968, 'total': 11873, 'HasAns_exact': 64.962520003369, 'HasAns_f1': 68.62438092233968, 'HasAns_total': 11873, 'best_exact': 64.962520003369, 'best_exact_thresh': 0.0, 'best_f1': 68.62438092233968, 'best_f1_thresh': 0.0}

max_answer_length = 100
{'exact': 64.92040764760381, 'f1': 68.59589159871524, 'total': 11873, 'HasAns_exact': 64.92040764760381, 'HasAns_f1': 68.59589159871524, 'HasAns_total': 11873, 'best_exact': 64.92040764760381, 'best_exact_thresh': 0.0, 'best_f1': 68.59589159871524, 'best_f1_thresh': 0.0}


-1 
{'exact': 76.88031668491536, 'f1': 80.14026054560372, 'total': 11873, 'HasAns_exact': 76.88031668491536, 'HasAns_f1': 80.14026054560372, 'HasAns_total': 11873, 'best_exact': 76.88031668491536, 'best_exact_thresh': 0.0, 'best_f1': 80.14026054560372, 'best_f1_thresh': 0.0}

-2
{'exact': 76.92242904068054, 'f1': 80.13062060284526, 'total': 11873, 'HasAns_exact': 76.92242904068054, 'HasAns_f1': 80.13062060284526, 'HasAns_total': 11873, 'best_exact': 76.92242904068054, 'best_exact_thresh': 0.0, 'best_f1': 80.13062060284526, 'best_f1_thresh': 0.0}
-3
{'exact': 76.89716162722142, 'f1': 80.10953189060857, 'total': 11873, 'HasAns_exact': 76.89716162722142, 'HasAns_f1': 80.10953189060857, 'HasAns_total': 11873, 'best_exact': 76.89716162722142, 'best_exact_thresh': 0.0, 'best_f1': 80.10953189060857, 'best_f1_thresh': 0.0}

{'exact': 75.56641118504169, 'f1': 80.15147116386775, 'total': 11873, 'HasAns_exact': 75.56641118504169, 'HasAns_f1': 80.15147116386775, 'HasAns_total': 11873, 'best_exact': 75.56641118504169, 'best_exact_thresh': 0.0, 'best_f1': 80.15147116386775, 'best_f1_thresh': 0.0}

{'exact': 75.56641118504169, 'f1': 80.15147116386775, 'total': 11873, 'HasAns_exact': 75.56641118504169, 'HasAns_f1': 80.15147116386775, 'HasAns_total': 11873, 'best_exact': 75.56641118504169, 'best_exact_thresh': 0.0, 'best_f1': 80.15147116386775, 'best_f1_thresh': 0.0}
