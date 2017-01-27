% Exercise #3 of Ensemble Learning: Ensemble Pruning via Individual Contribution Ordering (EPIC)

clear;

%% Prepares dataset.
dataSetName = 'ecoli' % (UCI)
[dataSet, Classes] = xlsread(dataSetName);
[ClassesIds ClassesLabels] = grp2idx(Classes);

%% Kfold to partition train and test data. K = 10
numPartitionsKFold = 10;
indices=crossvalind('Kfold', ClassesIds, numPartitionsKFold);

%% Number of weak learners in the pool
numClassifiers = [10 20 30 40 50 60 70 80 90 100];

for n = 1:size(numClassifiers,2)
	sprintf('Pool with %d classifiers...', numClassifiers(n))
	
	%% Kfold evaluation
	perfKfold = zeros(numPartitionsKFold,2);
	for i = 1:numPartitionsKFold
	    testIndices = (indices == i);
	    
	    % Select one partition to tune parameters of the pruning method
	    i2 = i + 1; 
	    if(i2 > numPartitionsKFold) 
	        i2 = 1; 
	    end
	    validationIndices = (indices == i2);

	    trainIndices = ~(testIndices | validationIndices);

	     % Train, pruning and test data sets
	    trainDataSet = dataSet(trainIndices,:);
	    trainClasses = ClassesIds(trainIndices,:);
	    validationDataSet = dataSet(validationIndices,:);
	    validationClasses = ClassesIds(validationIndices,:);
	    testDataSet = dataSet(testIndices,:);

	    %% Bagging is used to generate a pool of classifiers     

	    % Bootstrap sampling
	    [bootstatistic bootsamples] = bootstrp(numClassifiers(n),[], trainClasses);        

	    individualResultsTestData = zeros(size(testDataSet,1),numClassifiers(n));
	    individualResultsValidationData = zeros(size(validationDataSet,1),numClassifiers(n));
	    
	    % Classifiers training
	    for l = 1:numClassifiers(n)    
	        % Decision trees (CARTs)
	        cart = classregtree(trainDataSet(bootsamples(:,l),:), ...
	            ClassesIds(bootsamples(:,l),:), 'method', 'classification');
	        
	        % Save the accuracy of each tree in the validation and test data sets
	        individualResultsValidationData(:,l) = str2double(eval(cart, validationDataSet));
	        individualResultsTestData(:,l) = str2double(eval(cart, testDataSet));
	    end

	    %% Majority vote to classify the validation data
	    votes = zeros(size(validationDataSet,1),size(ClassesLabels,1));
	    for c=1:size(ClassesLabels,1)
	        votes(:,c) = sum(individualResultsValidationData == c, 2);
	    end

	    % Ordenates the votes in descending order
	    [ordenatedVotes IDX] = sort(votes, 2, 'descend');
	    resultsPool = IDX(:,1);
	    
	    %% Calculates the Individual Contribution for each classifier
	    IC = zeros(numClassifiers(n),1);
	    for l = 1:numClassifiers(n)
	        results_l = individualResultsValidationData(:,l);
	        alpha_l = (results_l == validationClasses) & (results_l ~= resultsPool);
	        beta_l = (results_l == validationClasses) & (results_l == resultsPool);
	        theta_l = results_l ~= validationClasses;
	        v_max = ordenatedVotes(:,1);
	        v_sec = ordenatedVotes(:,2);
	        v_l = zeros(size(validationDataSet,1),1);
	        v_correct = v_l;        
	        for x = 1:size(validationDataSet,1)
	            v_l(x) = votes(x,results_l(x));
	            v_correct(x) = votes(x,validationClasses(x));
	        end
	        IC(l) = sum(alpha_l.*(2*v_max - v_l) + beta_l.*(v_sec) + theta_l.*(v_correct - v_l - v_max));
	    end
	    
	    %% pruning the pool
	    p = 0.1;
	    [ordenatedIC classifiers_ID] = sort(IC,'descend');
	    ensembleSize = floor(numClassifiers(n) * p);
	    
	    %% Majority vote to classify the test data
	    votesPool = zeros(size(testDataSet,1),size(ClassesLabels,1));
	    votesEnsemble = votesPool;
	    for c=1:size(ClassesLabels,1)
	        votesPool(:,c) = sum(individualResultsTestData == c, 2);

	        % Pruned Ensemble Results
	        votesEnsemble(:,c) = sum(individualResultsTestData(:,classifiers_ID(1:ensembleSize)) == c, 2);
	    end
	    [v resultsPool] = max(votesPool,[],2);
	    [v resultsEnsemble] = max(votesEnsemble,[],2);

	    % Evaluates the pool and the pruned ensemble
	    perfKfold(i,1) = get(classperf(ClassesIds(testIndices,:),resultsPool),'ErrorRate');
	    perfKfold(i,2) = get(classperf(ClassesIds(testIndices,:),resultsEnsemble),'ErrorRate');
	end

% Stores the minimum error rate of Kfold evaluation
minPerfPoolTrees = min(perfKfold(:,1)); minPerfEnsembleTrees = min(perfKfold(:,2));
meanPerfPoolTrees = mean(perfKfold(:,1)); meanPerfEnsembleTrees = mean(perfKfold(:,2));
stdPerfPoolTrees = std(perfKfold(:,1)); stdPerfEnsembleTrees = std(perfKfold(:,2));

save(strcat(dataSetName,'_PerformanceOfThePolls_',int2str(numClassifiers(n)), ...
    '_Classifiers'), 'minPerfPoolTrees', 'meanPerfPoolTrees', 'stdPerfPoolTrees', ...
    'minPerfEnsembleTrees', 'meanPerfEnsembleTrees', 'stdPerfEnsembleTrees');

end

clear;