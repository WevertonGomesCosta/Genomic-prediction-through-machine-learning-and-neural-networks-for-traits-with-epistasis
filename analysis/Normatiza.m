function Matrizout= Normatiza(Matrizin,MatrizinBase,x0,x1)
valMax = max(MatrizinBase');
valMin = min(MatrizinBase');
[nl ncol] =size(Matrizin);
for k = 1:nl
    if valMax(k)-valMin(k) ~= 0
        Matrizout(k,:) = (x1+ ((Matrizin(k,:)-valMax(k))*(x1-x0))/ (valMax(k)-valMin(k)) );
    else
       Matrizout(k,:)= (x1+ ((Matrizin(k,:)-valMax(k))*(x1-x0)));    
    end
end
end