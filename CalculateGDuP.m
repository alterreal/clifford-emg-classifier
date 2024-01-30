function GD = CalculateGDuP(alpha,beta, XS_beta, YS_beta, n)
%CalculateGDuP is a function that computes the global distance (GD) between 
%a candidate gesture (alpha) and a training example, considering the sensors
%are n and are organized in a polygonal shape with n sides.
%
%INPUT
%       - alpha: candidate gesture (scaled values from each electrode channel)
%       - beta: training gesture (scaled values from each electrode channel)
%               (If beta is a set of training gestures, then GD will be a
%               vector containing the global distances between alpha and
%               each of the training gestures contained in beta)
%       - XS_beta, YS_beta: First shape of the training gesture
%       - n: number of polygon sides (i.e. number of sensors). Only 6 or 8.
%
%OUTPUT
%       - GD: global distance between the two gestures
%               (If beta is a set of training gestures, then GD will be a
%               vector containing the global distances between alpha and
%               each of the training gestures contained in beta)
%
%--------------------------------------------------------------------------
% Alexandre Calado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha_1 = [alpha(2:end) alpha(1)];

    delta_alpha_norm = alpha_1.^2 - alpha.*alpha_1+alpha.^2;               
    delta_alpha_norm(delta_alpha_norm == 0) = Inf;
    
    %For C code generation
    pwED = zeros(size(beta,1),1);
    fpwESd = zeros(size(beta,1),1);
    
    %Compute metrics according to the number of sides of the polygon
    switch n
        case 6
            %Compute the classical point-wise Euclidean distance
            pwED = sqrt(sum(abs(alpha - beta).^2, 2)/6);

            %pre-process alpha by calculating its first shape
            XS_alpha = (1/2).*((alpha.*alpha_1 - 2.*alpha.^2)./delta_alpha_norm);
            YS_alpha = (sqrt(3)/2).*(alpha.*alpha_1)./delta_alpha_norm;

            %Compute the First point-wise Euclidean shape distance
            fpwESd = (1/8)*sqrt(sum((XS_alpha - XS_beta).^2 + (YS_alpha - YS_beta).^2, 2));
        case 8
            %Compute the classical point-wise Euclidean distance
            pwED = sqrt(sum(abs(alpha - beta).^2, 2)/8);

            %pre-process alpha by calculating its first shape
            XS_alpha = (sqrt(2)/2).*((alpha.*alpha_1 - sqrt(2).*alpha.^2)./delta_alpha_norm);
            YS_alpha = (sqrt(2)/2).*(alpha.*alpha_1)./delta_alpha_norm;

            %Compute the First point-wise Euclidean shape distance
            fpwESd = (1/4)*sqrt(sum((XS_alpha - XS_beta).^2 + (YS_alpha - YS_beta).^2, 2));
    end

    %Compute the GD between alpha and beta
    GD =  fpwESd + pwED;
end

