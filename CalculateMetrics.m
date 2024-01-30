%Calculates precision, sensitivity, f1 score and specificity from the
%confusion matrix (cm)

function [cm, accuracy, abstention, precision, sensitivity,f1_score] = CalculateMetrics(Y, HypClass)
    

    m=length(Y);
    Class=unique(Y);
    
    cm=zeros(length(Class));
    m_abstention=zeros(1,length(Class));

    vec=zeros(m,1);
    count=0;

    for i=1:m

        if(Y(i)==HypClass(i))
            vec(i)=1;
        elseif(HypClass(i)==0)
            count=count+1;
        end

        if(HypClass(i)==0 && Y(i)~=0)
            m_abstention(Y(i))=m_abstention(Y(i))+1;
%         elseif(Y(i)~=0)
%             M(HypClass(i),Y(i))=M(HypClass(i),Y(i))+1;
        else
            cm(HypClass(i)+1,Y(i)+1)=cm(HypClass(i)+1,Y(i)+1)+1;
        end
    end

    accuracy=sum(vec)/(m-count)*100;
    abstention=count/m*100;

    precision = diag(cm)./sum(cm,2);
    sensitivity = diag(cm)./sum(cm,1)';
    
    precision(isnan(precision))=0;
    sensitivity(isnan(sensitivity))=0;
    
    f1_score = 2.*((precision.*sensitivity)./(precision + sensitivity));
    
    f1_score(isnan(f1_score))=0;
    
end

