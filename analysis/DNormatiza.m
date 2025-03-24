function Matrizout= DNormatiza(Matrizin,MatrizinBase,x0,x1)
valMax = max(MatrizinBase');
valMin = min(MatrizinBase');
[nl ncol] =size(Matrizin);

for k = 1:nl
      
    
    
     if valMax(k)-valMin(k) ~= 0
    %    Matrizout(k,:) = (x1+ ((Matrizin(k,:)-valMax(k))*(x1-x0))/ (valMax(k)-valMin(k)) );
        Matrizout(k,:) = valMax(k) +  ((Matrizin(k,:)-x1)* (valMax(k)-valMin(k)))/(x1-x0);
        
    else
         Matrizout(k,:) = valMax(k) +  ((Matrizin(k,:)-x1))/(x1-x0);
     end
    
     
     
end
end