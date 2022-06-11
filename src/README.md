This folder has the following structure:
<pre>
└── src				<- folder containing project source code
    ├── data			<- scripts to download/generate data
    │ 	├──stratified_sampling
    │	├──extract_audio
    │	├──extract_random_audio
    │ 	├──shuffle
    │ 	├──save_json
    │ 	├──import_data_dictionary
    │ 	├──intersection
    │ 	├──extract_set_list_from_text
    │ 	├──fast_default_data_splitting
    │ 	├──length_sanity_check
    │ 	├──add_random_noise
    │ 	├──envelope_coefficients
    │ 	├──set_dark_theme
    │	├──plot_audio
    │	├──plot_attention
    │	└──plots_attention_dataset
    │
    ├── features		<- scripts for feature extraction
    │ 	├──mfcc_feature_extraction
    │ 	├──energy
    │ 	├──short_time_energy
    │ 	├──zero_crossings
    │ 	├──mean_energies
    │ 	├──root_mean_square_energy
    │ 	└──log_specgram
    │
    │
    └── model			<- scripts for training, evaluation and prediction
	  ├──reshape
	  ├──normalize
    	  ├──plot_history
    	  ├──plot_confusion_matrix
    	  ├──wrong_predictions
      	├──rnn_architecture
      	└──att_rnn_architecture

</pre>
