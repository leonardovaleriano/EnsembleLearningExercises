% Exercise #1 of Ensemble Learning: Bagging Evaluation

clear;

% Experiments with two datasets: Ecoli and Fertility (UCI)
for t=1:2

% Datasets.
if(t == 1)
    dataSetName = 'ecoli'
else
    dataSetName = 'fertility'
end

[dataSet, Classes] = xlsread(dataSetName);
[LabelsIds Labels] = grp2idx(Classes);

%% Kfold to partition train and test data. K = 10
numPartitionsKFold = 10;
indices=crossvalind('Kfold', LabelsIds, numPartitionsKFold);

%% Number of weak learners in the pool and how many kind of weak learners
numClassifiers = [3 5 10 30 50 100];
typeOfClassifiers = 3;

for n = 1:size(numClassifiers,2)
	sprintf('Pool with %d classifiers...', numClassifiers(n))

	%% Kfold evaluation
	perfKfold = zeros(numPartitionsKFold,typeOfClassifiers);
	for i = 1:numPartitionsKFold
	    
	    % Train and test partitions
	    testIndices = (indices == i); 
	    trainIndices = ~testIndices;
	    
	    %% Bagging is used to generate a pool of classifiers      
	    
	    % Bootstrap sampling
	    trainDataSet = dataSet(trainIndices,:);
	    trainClasses = LabelsIds(trainIndices,:);
	    testDataSet = dataSet(testIndices,:);
	    [bootstatistic bootsamples] = bootstrp(numClassifiers(n),[], trainClasses);        
	    
	    %% Pool of classifiers
	    individualResultsOfPool = zeros(size(testDataSet,1),numClassifiers(n),typeOfClassifiers);
	    
	    % Evaluation of different kinds of weak learners: Decision Tree, kNN and MLP (10 neurons) 
	    for l = 1:numClassifiers(n)

	        % Decision trees (CARTs)
	        cart = classregtree(trainDataSet(bootsamples(:,l),:), LabelsIds(bootsamples(:,l),:), 'method', 'classification');        
	        individualResultsOfPool(:,l,1) = str2double(eval(cart, testDataSet));
	        
	        % kNNs (k-Nearest Neighbors)
	        individualResultsOfPool(:,l,2) = knnclassify(testDataSet, trainDataSet(bootsamples(:,l),:), LabelsIds(bootsamples(:,l),:), 5);
	        
	        % MLP (Multi layer perceptron)
	        net = patternnet(10);
	        net.trainParam.showWindow = false;
	        net = train(net,trainDataSet(bootsamples(:,l),:)', ind2vec(LabelsIds(bootsamples(:,l),:)'));
	        individualResultsOfPool(:,l,3) = vec2ind(net(testDataSet'))';
	    end

	    %% Majority vote to classify the test data
	    for k = 1:typeOfClassifiers
	        votes = zeros(size(testDataSet,1),size(Labels,1));
	        for c=1:size(Labels,1)
	            votes(:,c) = sum(individualResultsOfPool(:,:,k) == c, 2);
	        end

	        % Results with majority vote
	        [v results] = max(votes,[],2);
	        
	        % Ensemble loss evaluation
	        perfKfold(i,k) = get(classperf(LabelsIds(testIndices,:),results),'ErrorRate');
	    end        
	end

	% Stores the minimum error rate of Kfold evaluation
	perfPoolOfTrees = min(perfKfold(:,1));
	perfPoolOfKNNs = min(perfKfold(:,2));
	perfPoolOfMLPs = min(perfKfold(:,3));
	save(strcat(dataSetName,'_PerformanceOfThePolls_',int2str(numClassifiers(n)),'_Classifiers'), 'perfPoolOfTrees', 'perfPoolOfKNNs', 'perfPoolOfMLPs');
	end
	clear;
end
