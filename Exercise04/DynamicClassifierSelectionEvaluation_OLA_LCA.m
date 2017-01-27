clear;

%% Exercise #4 of Ensemble Learning: Dynamic Classifier Selection using the Overall Local Accuracy (OLA) and the Local Class Accuracy (LCA)

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
	sprintf('Pool with %d classifiers...', numClassifiers(n)

	%% Kfold evaluation
	perfKfold = zeros(numPartitionsKFold,3);
	for i = 1:numPartitionsKFold
	    testIndices = (indices == i);

	    % Select one partition for validation
	    i2 = i + 1; 
	    if(i2 > numPartitionsKFold) 
	        i2 = 1; 
	    end

	    validationIndices = (indices == i2);
	    trainIndices = ~(testIndices | validationIndices);
	    
	    % Train, validation and test data sets
	    trainDataSet = dataSet(trainIndices,:);
	    trainClasses = ClassesIds(trainIndices,:);
	    validationDataSet = dataSet(validationIndices,:);
	    validationClasses = ClassesIds(validationIndices,:);
	    testDataSet = dataSet(testIndices,:);

	    %% Bagging is used to generate a pool of classifiers  
	    
	    % Bootstrap sampling 
	    [bootstatistic bootsamples] = bootstrp(numClassifiers(n),[], trainClasses);        
	    
	    individualResultsTestData = zeros(size(testDataSet,1),numClassifiers(n));

	    % Classifiers training
	    pool = cell(numClassifiers(n));
	    for l = 1:numClassifiers(n)    
	      
	        % Decision trees (CARTs)
	        cart = classregtree(trainDataSet(bootsamples(:,l),:), ...
	            ClassesIds(bootsamples(:,l),:), 'method', 'classification');
	      
	        pool(l) = { cart };
	        individualResultsTestData(:,l) = str2double(eval(cart, testDataSet));
	    end

	    %% Dynamic Classifier Selection
	    % Finds dynamically the competence region using k-NN, k=5
	    knn = 5;
	    neighbors = knnsearch(validationDataSet, testDataSet, 'K', knn);
	    
	    % Number of classifiers chosed for the competence region
	    c_number = 5;
	    competenceRegionOLAResultsTestData=zeros(size(testDataSet,1),c_number);
	    competenceRegionLCAResultsTestData=zeros(size(testDataSet,1),c_number);
	    for k = 1:size(testDataSet,1)
	        competenceRegion = validationDataSet(neighbors(k,:),:);
	        competenceRegionClasses = validationClasses(neighbors(k,:));
	        competenceRegionCorrectRateOLA = zeros(numClassifiers(n),1);
	        competenceRegionCorrectRateLCA = zeros(numClassifiers(n),1);
	        for l = 1:numClassifiers(n)
	            answers = str2double(eval(pool{l},competenceRegion));
	    
	            %% OLA Precision            
	            competenceRegionCorrectRateOLA(l) = sum(answers == competenceRegionClasses) ...
	                / knn;
	    
	            %% LCA Precision
	            c = str2double(eval(pool{l}, testDataSet(k,:)));
	            c_indices = competenceRegionClasses == c;
	            if(~isempty(c_indices))
	                competenceRegionCorrectRateLCA(l) = sum(answers(c_indices) == c) / ...
	                    size(c_indices,1);
	            end
	        end
	    	
	    	%% Classifiers selection
	        [bestCorrectRates IDX_OLA] = sort(competenceRegionCorrectRateOLA, 'descend');
	        [bestCorrectRates IDX_LCA] = sort(competenceRegionCorrectRateLCA, 'descend');
	        competenceRegionClassifiersOLA = pool(IDX_OLA(1:c_number));
	        competenceRegionClassifiersLCA = pool(IDX_LCA(1:c_number));
	        
	        %% Evaluation of selected classifiers in the test data set
	        for l = 1:c_number
	            competenceRegionOLAResultsTestData(k,l) = ...
	                str2double(eval(competenceRegionClassifiersOLA{l}, testDataSet(k,:)));
	            competenceRegionLCAResultsTestData(k,l) = ...
	                str2double(eval(competenceRegionClassifiersLCA{l}, testDataSet(k,:)));
	        end
	    end
	    
	    %% Uses majority vote to classify the test data
	    votesPool = zeros(size(testDataSet,1),size(ClassesLabels,1));
	    votesOLA = votesPool;
	    votesLCA = votesPool;
	    for c=1:size(ClassesLabels,1)
	        votesPool(:,c) = sum(individualResultsTestData == c, 2);
	        votesOLA(:,c) = sum(competenceRegionOLAResultsTestData == c, 2);
	        votesLCA(:,c) = sum(competenceRegionLCAResultsTestData == c, 2);
	    end	
	    [v resultsPool] = max(votesPool,[],2);
	    [v resultsOLA] = max(votesOLA,[],2);
	    [v resultsLCA] = max(votesLCA,[],2);
	    
	    % Evaluates Bagging and dynamic selection ensemble
	    perfKfold(i,1) = get(classperf(ClassesIds(testIndices,:),resultsPool),'ErrorRate');
	    perfKfold(i,2) = get(classperf(ClassesIds(testIndices,:),resultsOLA),'ErrorRate');
	    perfKfold(i,3) = get(classperf(ClassesIds(testIndices,:),resultsLCA),'ErrorRate');
	end

% Stores the minimum error rate of Kfold evaluation
minPerfPoolTrees = min(perfKfold(:,1)); minPerfOLA = min(perfKfold(:,2));
minPerfLCA = min(perfKfold(:,3)); meanPerfPoolTrees = mean(perfKfold(:,1)); ...
meanPerfOLA = mean(perfKfold(:,2)); meanPerfLCA = mean(perfKfold(:,3));
stdPerfPoolTrees = std(perfKfold(:,1)); stdPerfOLA = std(perfKfold(:,2));
stdPerfLCA = std(perfKfold(:,3));

save(strcat(dataSetName,'_PerformanceOfThePolls_',int2str(numClassifiers(n)), ...
    '_Classifiers'), 'minPerfPoolTrees', 'meanPerfPoolTrees', 'stdPerfPoolTrees', ...
    'minPerfOLA', 'meanPerfOLA', 'stdPerfOLA', 'minPerfLCA', 'meanPerfLCA', 'stdPerfLCA');
end

clear;