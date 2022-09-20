function ImportanciaRM= ImportanciaGWS(rede,xxin,yyin)
yyrede = sim(rede,xxin);
% valores iniciais
matriz = [yyrede' yyin'];
erro =yyrede'- yyin';
EQMT= mse(erro);
[r,p] = corrcoef(matriz);
R2T = r(1,2)*r(1,2);

fprintf('Import�ncia dos marcadores - An�lise com todos caracteres\n');
fprintf('R2 porc.  :  %f\n',100*R2T);
fprintf('EQM  :  %f\n',EQMT);
ImportanciaRM = [0 R2T EQMT];
fprintf('\n');

%% M�todo 1
[nvar nind] = size(xxin);
%fprintf('M�todo 1: Zerar o valor da vari�vel\n');
%Importancia1= [];
% for i=1:nvar
%     xx = xxin;
%     xx(i,:) = zeros(1,nind);
%     yyrede = sim(rede,xx);
%     matriz = [yyrede' yyin'];
%     erro =yyrede'- yyin';
%     EQM= mse(erro);
%     [r,p] = corrcoef(matriz);
%     R2 = r(1,2)*r(1,2);
%     IRR2 = (R2T - R2)/R2T;
%     ImportanciaRM = [ImportanciaRM; i R2 EQM];
%     Importancia1= [Importancia1; i R2 EQM];
%     fprintf('Variavel:  %d   R2: %f    EQM:  %f  IRR2:  %f\n',i, 100*R2, EQM, IRR2 );
%     
% end

%% M�todo 2
fprintf('\n');
fprintf('M�todo : casualiza��o da vari�vel\n');
Importancia2= [];
for i=1:nvar
    xx = xxin;
    
    embaralha = randperm(nind);
    xx(i,:) = xx(i,embaralha);
    yyrede = sim(rede,xx);
    matriz = [yyrede' yyin'];
    erro =yyrede'- yyin';
    EQM= mse(erro);
    [r,p] = corrcoef(matriz);
    R2 = r(1,2)*r(1,2);
    IRR2 = (R2T - R2)/R2T;
    ImportanciaRM = [ImportanciaRM; i R2 EQM];
    %fprintf('Variavel:  %d   R2: %f    EQM:  %f  IRR2:  %f\n',i, 100*R2, EQM, IRR2 );
    
    
    Importancia2= [Importancia2; i R2 EQM];
end

fprintf('Valores de R� : \n');
saidagws= Importancia2(:,2)'
fprintf('Marcadores mais importantes (50) com base no  R� : \n');
[B, IX] = sort(abs(saidagws), 'descend');

IX(1,1:50)

fprintf('Valores de EQM : \n');
saidagws= Importancia2(:,3)'
[B, IX] = sort(abs(saidagws), 'descend');
fprintf('Marcadores mais importantes(50) com base no EQM : \n');
IX(1,1:50)
figure(10)
%  subplot(4,1,1)
% hold on
% X1 = Importancia1(:,2);
% bar(X1,'r')
% legend('R�')
% ylabel('Y')
% title('M�todo 1 - Zerar vari�vel')
% hold off
% 
% subplot(4,1,2)
% hold on
% X1 = Importancia1(:,3);
% bar(X1,'k')
% legend('EQM')
% ylabel('Y')
% title('M�todo 1 - Zerar vari�vel')
% hold off
 
subplot(2,1,1)
hold on
X1 = Importancia2(:,2);
bar(X1,'r')
%legend('R�')
ylabel('Y')
title('M�todo  - Casualizar vari�vel R�')
hold off


subplot(2,1,2)
hold on
X1 = Importancia2(:,3);
bar(X1,'k')
%legend('EQM')
ylabel('Y')
title('M�todo  - Casualizar vari�vel EQM')
hold off

 saveas(gcf,'c:\dados\ScriptsM\Saidas\grafico0.bmp', 'bmp')

end                       