function Matrizout= NNkmeans(Matrizin,NNmax)
eixoxk = 2:NNmax;
eixoyk = [];
for i = 2:NNmax
    nk=i;
[idx,C,sumd]= kmeans(Matrizin',nk);
eixoyk(i-1) =sum(sumd);
end
 figure(20);
         hold on
         plot(eixoxk,eixoyk,'k-');
         texto = ['Arquivo de treinamento - k-means' ];
         title(texto)
         xlabel('Valor de k')
         ylabel('SQ dentro')
         hold off
end