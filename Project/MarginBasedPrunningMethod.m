clear;

%% Project of Ensemble Learning: Replication of results from article "Margin-based ordered aggregation for ensemble pruning (Pattern Recognition Letters - 2013)"

%% Prepares datasets.
dataSetNames = {'connect-4', 'glass', 'kr-vs-kp', 'letter-recognition', 'optdigits', ...
    'pendigits', 'pima-indians-diabetes', 'tic-tac-toe', 'waveform', 'winequality-red' };

dataSetCategoricalVectors = { 1:42, [], 1:36, [], [], [], [], 1:9, [], [] };

% Number of instances for training, validation and test for each data set
dataSetInstances = { [2000 2000 2000], [72 71 71], [1065 1065 1065], [2000 2000 2000], ...
    [1000 1000 1000], [2000 2000 2000], [256 256 256], [310 310 310], [1000 1000 1000], [533 533 533] };

% Puts noise in the original data 
experimentType = { 'normal', '5_percent_noise', '10_percent_noise' };

%% Experiments
for type=1:size(experimentType,2)
	
	for d=1:size(dataSetNames,2)
		%% Loads the data set
		strcat('Processing data set:', dataSetNames{d})
		strcat('Experiment type:', experimentType{type})
		[dataSetInicial, Classes] = xlsread(strcat('Data Sets\',dataSetNames{d}));
		if(isempty(Classes))
		    Classes = dataSetInicial(:,size(dataSetInicial,2));
		end
		[ClassesIdsInicial ClassesLabels] = grp2idx(Classes);
		
		%% Defines the initial number of classifiers in the pool
		numClassifiers = [ 1 51 101 151 201 251 301 351 401 451 501 ];
		perfDataSet = zeros(size(numClassifiers,2),3,2); errorRateDataSet = zeros(size(numClassifiers,2),size(ClassesLabels,1),3);
		M_logDataSet = zeros(size(numClassifiers,2),2); KW_logDataSet = zeros(size(numClassifiers,2),3);
		for n = 1:size(numClassifiers,2)
		    sprintf('Pool with %d classifiers...', numClassifiers(n))
			
			%% Number of times that each experiment is executed
		    nTimes = 10; perfNTimes = zeros(nTimes,3); errorRate = zeros(nTimes,size(ClassesLabels,1),3);
		    numInstances = dataSetInstances{d}; M_log = zeros(nTimes,1); M_Accuracy_log = zeros(nTimes,1);
		    KW_Pool_log = zeros(nTimes,1); KW_Margin_log = zeros(nTimes,1); KW_Accuracy_log = zeros(nTimes,1);
		    for i = 1:nTimes        
		
				% Creates the training partition
		        [ignoreIndices trainIndices]=crossvalind('LeaveMOut', length(ClassesIdsInicial), numInstances(1));
		        trainDataSet = dataSetInicial(trainIndices,:); trainClassesIds = ClassesIdsInicial(trainIndices,:);   
		        remainingDataSet = dataSetInicial(ignoreIndices,:); remainingClassesIds = ClassesIdsInicial(ignoreIndices,:);
				
				% Creates the validation partition    
		        [ignoreIndices validationIndices] = crossvalind('LeaveMOut', length(remainingClassesIds), numInstances(2));
		        validationDataSet = remainingDataSet(validationIndices,:); validationClassesIds = remainingClassesIds(validationIndices,:);
		        remainingDataSet = remainingDataSet(ignoreIndices,:); remainingClassesIds = remainingClassesIds(ignoreIndices,:);
				
				% Creates the test partition
		        [ignoreIndices testIndices] = crossvalind('LeaveMOut', length(remainingClassesIds), numInstances(3));
		        testDataSet = remainingDataSet(testIndices,:); testClassesIds = remainingClassesIds(testIndices,:);
		
				% Put noise in the classes labels
		        if(type == 2)
		            noiseLevel = 0.05;
		        elseif(type == 3)
		            noiseLevel = 0.1;
		        end

		        % Hold out evaluation
		        if(type ~= 1)
		            [normalIndices noisedIndices] = crossvalind('HoldOut', size(trainClassesIds,1), noiseLevel);
		            trainClassesIds(noisedIndices) = randi(size(ClassesLabels,1),sum(noisedIndices),1);
		            [normalIndices noisedIndices] = crossvalind('HoldOut', size(validationClassesIds,1), noiseLevel);
		            validationClassesIds(noisedIndices) = randi(size(ClassesLabels,1),sum(noisedIndices),1);
		            [normalIndices noisedIndices] = crossvalind('HoldOut', size(testClassesIds,1), noiseLevel);
		            testClassesIds(noisedIndices) = randi(size(ClassesLabels,1),sum(noisedIndices),1);
		        end

				%% Bagging is used to generate a pool of classifiers     
		        [bootstatistic bootsamples] = bootstrp(numClassifiers(n),[], trainClassesIds);
		        
		        individualResultsvalidationData = zeros(size(validationDataSet,1),numClassifiers(n));
		        individualResultsTestData = zeros(size(testDataSet,1),numClassifiers(n));        
		        
		        % Classifiers training
		        for l = 1:numClassifiers(n)
		            cart = classregtree(trainDataSet(bootsamples(:,l),:), trainClassesIds(bootsamples(:,l),:), ...
		                'method', 'classification', 'categorical', dataSetCategoricalVectors{d});
		            individualResultsvalidationData(:,l) = str2double(eval(cart, validationDataSet));
		            individualResultsTestData(:,l) = str2double(eval(cart, testDataSet));
		        end    

				%% Majority vote to classify the validation data
		        votes = zeros(size(validationDataSet,1),size(ClassesLabels,1));
		        for c=1:size(ClassesLabels,1)
		            votes(:,c) = sum(individualResultsvalidationData == c, 2);
		        end

				% Ordenates the votes in descending order
		        [ordenatedVotes IDX] = sort(votes, 2, 'descend');
		
				% Calculates the margins of each instance and classifier
		        margin = (-1/size(validationDataSet,1))*log((ordenatedVotes(:,1) - ordenatedVotes(:,2))/numClassifiers(n))';
		        marginClassifier = margin*(individualResultsvalidationData == repmat(validationClassesIds,1,numClassifiers(n)));
		        
		        % Classifiers ordering
		        [ordenatedMargin classifiers_ID] = sort(marginClassifier,'descend');
		        perfvalidation = zeros(1,numClassifiers(n));
		        
		        % Classifiers evaluation
		        for l=1:numClassifiers(n)
		            perfvalidation(l) = get(classperf(validationClassesIds,individualResultsvalidationData(:,l)),'CorrectRate');
		        end
		        [ordenatedAccuracy classifiers_ID_AccuracyMethod] = sort(perfvalidation, 'descend');
		
				% Defines M value to prune the pool through the validation data
		        perfvalidation = zeros(1,numClassifiers(n)); perfvalidationAccuracyMethod = perfvalidation;
		        for l=1:numClassifiers(n)
		            votes = zeros(size(validationDataSet,1),size(ClassesLabels,1)); votesAccuracyMethod = votes;
		            for c=1:size(ClassesLabels,1)
		                votes(:,c) = sum(individualResultsvalidationData(:,classifiers_ID(1:l)) == c, 2);
		                votesAccuracyMethod(:,c) = sum(individualResultsvalidationData(:,classifiers_ID_AccuracyMethod(1:l)) == c, 2);
		            end
		            [v resultsEnsemble] = max(votes,[],2);
		            [v resultsEnsembleAccuracyMethod] = max(votesAccuracyMethod,[],2);
		            perfvalidation(l) = get(classperf(validationClassesIds,resultsEnsemble),'CorrectRate');
		            perfvalidationAccuracyMethod(l) = get(classperf(validationClassesIds,resultsEnsembleAccuracyMethod),'CorrectRate');
		        end
		        [v M] = max(perfvalidation); [v M_AccuracyMethod] = max(perfvalidationAccuracyMethod);
		        M_log(i) = M; M_Accuracy_log(i) = M_AccuracyMethod;
		
				%% Majority vote to classify the test data
		        votesPool = zeros(size(testDataSet,1),size(ClassesLabels,1));
		        votesEnsemble = votesPool; votesEnsembleAccuracyMethod = votesPool;
		        for c=1:size(ClassesLabels,1)
		            votesPool(:,c) = sum(individualResultsTestData == c, 2);
		            votesEnsemble(:,c) = sum(individualResultsTestData(:,classifiers_ID(1:M)) == c, 2);
		            votesEnsembleAccuracyMethod(:,c) = sum(individualResultsTestData(:,classifiers_ID_AccuracyMethod(1:M_AccuracyMethod)) == c, 2);
		        end
		        [v resultsPool] = max(votesPool,[],2); 
		        [v resultsEnsemble] = max(votesEnsemble,[],2);
		        [v resultsEnsembleAccuracyMethod] = max(votesEnsembleAccuracyMethod,[],2);
		
				%% Ensembles evaluation using Kohavi-Wolpert variance (KW) diversity measure
		        KW = 1 / (size(testClassesIds,1) * numClassifiers(n)^2);
		        l_x = diag(votesPool(:,testClassesIds)); KW_Pool_log(i) = KW * sum( l_x .* (numClassifiers(n) - l_x) );
		        l_x = diag(votesEnsemble(:,testClassesIds)); KW_Margin_log(i) = KW * sum( l_x .* (numClassifiers(n) - l_x) );
		        l_x = diag(votesEnsembleAccuracyMethod(:,testClassesIds)); KW_Accuracy_log(i) = KW * sum( l_x .* (numClassifiers(n) - l_x) );
		
				%% Ensembles evaluation using correct rate metric
		        perfNTimes(i,1) = get(classperf(testClassesIds,resultsPool),'CorrectRate');
		        perfNTimes(i,2) = get(classperf(testClassesIds,resultsEnsemble),'CorrectRate');
		        perfNTimes(i,3) = get(classperf(testClassesIds,resultsEnsembleAccuracyMethod),'CorrectRate');
		
				%% Ensembles evaluation using error rate metric by class      
		        for c=1:size(ClassesLabels,1)
		            c_instances = testClassesIds == c;
		            errorRate(i,c,1) = sum(resultsPool ~= testClassesIds & c_instances)/ sum(c_instances);
		            errorRate(i,c,2) = sum(resultsEnsemble ~= testClassesIds & c_instances)/ sum(c_instances);
		            errorRate(i,c,3) = sum(resultsEnsembleAccuracyMethod ~= testClassesIds & c_instances)/ sum(c_instances);
		        end
		    end

			%% Average results from the nTimes evaluation
			perfDataSet(n,:) = [ mean(perfNTimes) std(perfNTimes) ];
			errorRateDataSet(n,:,:) = mean(errorRate,1);
			M_logDataSet(n,:) = [ mean(M_log) mean(M_Accuracy_log) ];
			KW_logDataSet(n,:) = [ mean(KW_Pool_log) mean(KW_Margin_log) mean(KW_Accuracy_log) ];
			save(strcat('Results\',dataSetNames{d},'_Results_',experimentType{type}),'perfDataSet','errorRateDataSet','M_logDataSet','KW_logDataSet');
		end
	end

end
clear;