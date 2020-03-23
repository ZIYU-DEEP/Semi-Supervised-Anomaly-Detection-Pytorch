python main.py -ln rec_unsupervised -le rec_eval -nt rec -op rec_unsupervised -nf downtown -af _ -rt /net/adv_spectrum/torch_data_deepsad/100 -gpu 1 --n_epochs 160 - gpu 1
python main.py -ln rec -le rec_eval -nt rec -op rec_unsupervised -nf downtown -af downtown_sigOver_10ms -rt /net/adv_spectrum/torch_data_deepsad/100 --n_epochs 160 -gpu 1
python main.py -ln forecast_unsupervised -le forecast_eval -nt lstm_stacked -op forecast_unsupervised -nf downtown -af _ -rt /net/adv_spectrum/torch_data -gpu 1
python main.py -ln forecast -le forecast_eval -nt lstm_stacked -op forecast_exp -nf downtown -af downtown_sigOver_10ms -rt /net/adv_spectrum/torch_data -gpu 1
python main.py -ln rec_unsupervised -le rec_eval -nt rec -op rec_unsupervised -nf ryerson_train -af _ -rt /net/adv_spectrum/torch_data_deepsad/100 -gpu 1 --n_epochs 160 - gpu 1
python main.py -ln rec -le rec_eval -nt rec -op rec_unsupervised -nf ryerson_train -af ryerson_ab_train_sigOver_10ms -rt /net/adv_spectrum/torch_data_deepsad/100 --n_epochs 160 -gpu 1
python main.py -ln forecast_unsupervised -le forecast_eval -nt lstm_stacked -op forecast_unsupervised -nf ryerson_train -af _ -rt /net/adv_spectrum/torch_data -gpu 1
python main.py -ln forecast -le forecast_eval -nt lstm_stacked -op forecast_exp -nf ryerson_train -af ryerson_ab_train_sigOver_10ms -rt /net/adv_spectrum/torch_data -gpu 1
