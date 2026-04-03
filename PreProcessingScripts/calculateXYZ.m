function [XYZ] = calculateXYZ(Range)
    % Setup Parameters
    fov = 3*[1 1]; % [az el] adjust this to match sensor FOV
    sensorLen = 128;

    % Initialize matrices
    az = linspace(-fov(1)/2,fov(1)/2,sensorLen);
    el = linspace(-fov(2)/2,fov(2)/2,sensorLen);
    az_matrix = repmat(az,[numel(el) 1]); %flip az as well to get same orientation as raw images
    el_matrix= repmat(flip(el)',[1 numel(az)]);
    
    % Calculate cartesian coordinates from azimuth and elevation
    [x_matrix,y_matrix,z_matrix] = sph2cart(az_matrix*pi/180,el_matrix*pi/180,Range);   
    XYZ = cat(3,x_matrix,y_matrix,z_matrix);
end