function [OutVehicleDS] = balanceDataset(vehicleDataset, maxCount)
    %%% Fix issue when using small number cap e.g. 100
    
    % Convert Table to Label Count Matrix
    CELL = table2cell(vehicleDataset);
    COUNT = cellfun(@(x) size(x,1), CELL, 'UniformOutput',false);
    CountMAT = cell2mat(COUNT(:,2:end));
    
    % Count number of labels per row
    rowCount = sum(CountMAT, 2);
    
    % Take every fourth frame from multivehicle videos
    mvLOG = rowCount > 1;
    mvInd = find(mvLOG);
    mvIndGood = mvInd(1:4:end);
    
    % Count number of labels per class
    classCount = sum(CountMAT(mvIndGood,:));
    
    % Take enough frames from single vehicle videos to reach maxCount for each class
    missingCount = maxCount - classCount;
    
    % Run for loop to take remaining frames from each class
    svIndGood = [];
    for ii = 1:width(CountMAT)
        svInd = find(CountMAT(:,ii) >= 1 & ~mvLOG);
        skipInd = round(linspace(1,length(svInd), missingCount(ii)));
        svIndGood = [svIndGood; svInd(skipInd)];    
    end
    
    % Create output training table
    OutVehicleDS = vehicleDataset([mvIndGood; svIndGood], :);
end
