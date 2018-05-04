addpath('../borrowed code/f_PlotEEG_BrainNetwork/f_PlotEEG_BrainNetwork')
wt = [72657.96797943115 72386.41053390503 74135.6166381836 72147.28018951416 70802.43060302734 72689.00830841064 73765.27107620239 73431.64393234253 70482.14329528809 70595.47065353394 70864.51742553711 74245.80281448364 74752.28070449829 72306.3685760498 71307.25164413452 72284.97057723999 75312.37509536743 73113.31763076782 70970.39012145996 71496.54410552979 74028.08048629761 77176.96394729614 72981.96081924438 72811.82777404785 73758.03799438477 79459.52325057983 78561.44002532959 76023.90644454956 76883.95377731323 78954.46450042725 77894.11158752441];
wt = wt / (18*10*10*30*64);

t = 1:31;
t = [t' t' ones(size(t'))];
f_PlotEEG_BrainNetwork(31, t, 'w_intact', wt, 'n_nn2nx', 'ncb')