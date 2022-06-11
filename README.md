# Keyword-Spotting-Recognition-through-state-of-art-neural-architectures

In this repo is presented the work by Francesco Ferretto and Federica Latora on the study of 
Deep Learning applications for Audio Speech Recognition (ASR). 

It follows the tree structure of the project: 
<pre>
├── data
│   ├── binaries		<- .npy data
│   ├── processed		<- processed data (features)
│   └── raw			<- raw unedited data
│       ├── data_documentation	<- original data details
│       ├── dataset_v2_reduced  <- reduced dataset (10 classes)
│       │     ├── down
│       │     ├── go
│       │     ├── left
│       │     ├── no
│       │     ├── off
│       │     ├── on
│       │     ├── right
│       │     ├── stop
│       │     ├── up
│       │     └── yes   
│       └── dataset_v2  <- complete dataset (35 classes)
│             ├── backward
│             ├── bed
│             ├── bird
│             ├── cat
│             ├── dog
│             ├── down
│             ├── eight
│             ├── five
│             ├── follow
│             ├── forward
│             ├── four
│             ├── go
│             ├── happy
│             ├── house
│             ├── learn
│             ├── left
│             ├── marvin
│             ├── nine
│             ├── no
│             ├── off
│             ├── on
│             ├── one
│             ├── right
│             ├── seven
│             ├── sheila
│             ├── six
│             ├── stop
│             ├── three
│             ├── tree
│             ├── two
│             ├── up
│             ├── visual
│             ├── wow
│             ├── yes
│             └── zero
├── models			<- notebooks for experimental purposes for modeling (comparison, ensemble, ...)
├── notebooks			<- prototyping notebooks for EDA, feature selection, ...
└── src				<- folder containing project source code
    ├── data			<- scripts to download/generate data
    ├── features		<- scripts for feature extraction
    └── model			<- scripts for training, evaluation and prediction
	
</pre>

