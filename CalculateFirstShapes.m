function [XS, YS] = CalculateFirstShapes(beta, n)
%CalculateFirstShapes is a function that computes the first shapes XS and
%YS of sample(s) beta, considering n features that can be organized as an
%polygon of n sides (Only 6 or 8).
%--------------------------------------------------------------------------
% Alexandre Calado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    beta_1 = [beta(:,2:end) beta(:,1)];

    delta_beta_norm = beta_1.^2 - beta.*beta_1 + beta.^2;
    delta_beta_norm(delta_beta_norm == 0) = Inf;
    
    XS = zeros(size(beta));
    YS = zeros(size(beta));
    
    switch n
        case 6        
            XS = (1/2).*((beta.*beta_1 - 2.*beta.^2)./delta_beta_norm);
            YS = (sqrt(3)/2).*(beta.*beta_1)./delta_beta_norm;
        case 8 
            XS = (sqrt(2)/2).*((beta.*beta_1 - sqrt(2).*beta.^2)./delta_beta_norm);
            YS = (sqrt(2)/2).*(beta.*beta_1)./delta_beta_norm;
    end

end

