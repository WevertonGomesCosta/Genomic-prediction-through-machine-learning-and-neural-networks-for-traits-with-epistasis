%% ************************************************************************
% ***************************** 1. Inicialização *************************
% *************************************************************************
clear,  close all, clc;
warning off
rand('seed', 1)

%% ************************************************************************
% ******************** 2. Leitura dos arquivos ****************************
% *************************************************************************
caminho = ('I:\Weverton\dados_qualificacao\Results_rbf\');
pathFenT = ('I:\Weverton\dados_qualificacao\PHEN_T\');
pathFenV = ('I:\Weverton\dados_qualificacao\PHEN_V\');
pathGenT = ('I:\Weverton\dados_qualificacao\GEN_T\');
pathGenV = ('I:\Weverton\dados_qualificacao\GEN_V\');


filesFenT = dir([pathFenT '*.txt']);
filesFenV = dir([pathFenV '*.txt']);
filesGenT = dir([pathGenT '*.txt']);
filesGenV = dir([pathGenV '*.txt']);

[nrow, ncol] = size(filesFenT);

%% ************************************************************************
% *********** 3. Declaração dos parâmetros iniciais ***********************
% *************************************************************************
EQMmin = 0.05;
Raioi = 20;
Raiof = 50;
Raiopasso = 10;
NNi = 15;
NNf = 30;
NNpasso = 5;
Display = NNi;

%% ************************************************************************
% *********** 4. Processamento da RBF otimizada *************************
% *************************************************************************
nvariavel = 18;

%%
for v = 1:nvariavel
R2trei = [];
R2val = [];
reqt = [];
reqv = [];
raioo=[];
nno=[];
varsel = v;
%%
for i = 1:nrow
   %%
    diary on
    diary([caminho 'parcial' num2str(varsel) num2str(i) '.txt']);
    dadosTy = importdata([pathFenT filesFenT(i).name]); 
    dadosTx = importdata([pathGenT filesGenT(i).name]); 
    dadosVy = importdata([pathFenV filesFenV(i).name]); 
    dadosVx = importdata([pathGenV filesGenV(i).name]); 
    %%    
    %************************************************************************
    % *********** 5. Montagem dos arquivos de entrada e saída *****************
    % *************************************************************************
    matrizfo = dadosTy(:,varsel)';
    saidasT = Normatiza(matrizfo,matrizfo,-1,1);
    
    matrizgo = dadosTx';
    entradasT  = Normatiza(matrizgo,matrizgo,-1,1);
   
    matrizfv = dadosVy(:,varsel)';
    saidasV  = Normatiza(matrizfv,matrizfo,-1,1);
    
    matrizgv= dadosVx';
    entradasV = Normatiza(matrizgv,matrizgo,-1,1);
    
    [nLin nCol] = size(entradasT);
    numEntrada= nLin;
    numIndividuo= nCol;
    
    [nLinv nColv] = size(entradasV);
    numEntradav= nLinv;
    numIndividuov= nColv;
    
    %% ************************************************************************
    % *********** 6. Processamento da RBF   ***********************************
    % *************************************************************************
    
    R2vmin=0;
    R2tmin =0;
    desempenho =[];
    perfvmin=999999999999999;
   
    
    for nn=NNi:NNpasso:NNf
        
        if R2vmin == 1
            break
        end
        
        for raio=Raioi:Raiopasso:Raiof
            
            if R2vmin == 1 
                break
            end
            
            net= newrb(entradasT,saidasT,EQMmin,raio,nn, Display);
            saidasRT= sim(net,entradasT);
            matriz = [saidasT',saidasRT'];
            [r,p] = corrcoef(matriz);
            R2t = r(1,2)*r(1,2);
            
            saidasRV = sim(net,entradasV);
            matriz = [saidasV',saidasRV'];
            [r,p] = corrcoef(matriz);
            R2v = r(1,2)*r(1,2);
            
            perfT = perform(net,saidasT,saidasRT);
            perfV = perform(net,saidasV,saidasRV);
            desempenho = [desempenho ;nn raio R2t R2v perfT perfV];
            
            if  perfV <= perfvmin
                perfvmin = perfV;
                R2tmin = R2t;
                R2vmin = R2v;
                minhanet = net;
                raiootimo=raio;
                nnotimo=nn;
            end
        end       
    end  
diary off
    %% ************************************************************************
    % *********** 7. Resultado Final  *****************************************
    % *************************************************************************
    
    %______________________________________________
    % 7a Informações iniciais
    %______________________________________________
    
    diary on
    diary([caminho 'final' num2str(varsel) num2str(i) '.txt'])
    fprintf('\n');
    fprintf('------------------------------------- \n');
    fprintf('Resultado FINAL   \n');
    fprintf('------------------------------------- \n');
    fprintf('Variável principal (Y) :  %d\n',varsel);
    fprintf('Número do Fold :  %d\n',i);
    fprintf('EQM :  %d\n',EQMmin);
    fprintf('Raio inicial :  %d\n',Raioi);
    fprintf('Raio final :  %d\n',Raiof);
    fprintf('Raio passo :  %d\n',Raiopasso);
    fprintf('Número de neurônios inicial :  %d\n',NNi);
    fprintf('Número de neurônios final :  %d\n',NNf);
    fprintf('Número de neurônios passo :  %d\n',NNpasso);
    
    
    %%______________________________________________
    % 7b Desempenho da RBF
    %______________________________________________
    raioo(i)= raiootimo; 
    nno(i)= nnotimo;
    fprintf('Raio ótimo :  %d\n',raiootimo);
    fprintf('Número de neurônios ótimo :  %d\n',nnotimo);
    
    %______________________________________________
    % 7c. Valores de R2 para treinamento e validação
    %______________________________________________
    
    fprintf('\n');
    fprintf('------------------------------------- \n');
    fprintf('Resultado do Treinamento: \n');
    fprintf('Entradas: %5.0f\n',numEntrada);
    fprintf('Indivíduos: %5.0f\n',numIndividuo);
    fprintf('--------------------------------------------\n');
    saidasRT = sim(minhanet,entradasT);
    matriz = [saidasT',saidasRT'];
    [r,p] = corrcoef(matriz);
    R2trei(i) = r(1,2)*r(1,2);
    y1  = DNormatiza(saidasRT ,matrizfo,-1,1);
    y2  = DNormatiza(saidasT ,matrizfo,-1,1);
    y1 = y1 - mean(y1) + mean(y2);
    EQMtx= mse(y1'-y2');
    reqt(i) = sqrt(EQMtx);
    
        
    fprintf('Resultado de Validação: \n');
    fprintf('Entradas: %5.0f\n',numEntradav);
    fprintf('Indivíduos: %5.0f\n',numIndividuov);
    fprintf('--------------------------------------------\n');
    saidasRV = sim(minhanet,entradasV);
    matriz = [saidasV',saidasRV'];
    [r,p] = corrcoef(matriz);
    R2val(i) = r(1,2)*r(1,2);
    y1  = DNormatiza(saidasRV ,matrizfo,-1,1);
    y2  = DNormatiza(saidasV ,matrizfo,-1,1);
    y1 = y1 - mean(y1) + mean(y2);
    EQMvx= mse(y1'-y2');
    reqv(i)= sqrt(EQMvx);
    fprintf('\n');

    %%
    %______________________________________________
    % 7d. Salvando os resultados
    %______________________________________________
    
    resultrbf = [R2trei(i) R2val(i) reqt(i) reqv(i) raioo(i) nno(i)];
    rownames = {['KFold' num2str(i)]};
    colnames = {'R2Trein', 'R2Val', 'REQMTrein', 'REQMVal', 'RaioOtimo', 'NeuronioOtimo'};   
    resultrbf = array2table(resultrbf, 'VariableNames', colnames,'RowNames', rownames)
        
     %%
    %______________________________________________
    % 8. Importância das variáveis
    %______________________________________________
    yyrede = sim(minhanet,entradasT);
    %______________________________________________
    % 8a. Parâmetros iniciais 
    %______________________________________________
    matriz = [yyrede' saidasT'];
    erro =yyrede'- saidasT';
    EQMT= mse(erro);
    [r,p] = corrcoef(matriz);
    R2T = r(1,2)*r(1,2);
    
    fprintf('Importância dos marcadores - Análise com todos caracteres\n');
    fprintf('R2 porc.  :  %f\n',100*R2T);
    fprintf('EQM  :  %f\n',EQMT);
    fprintf('\n');
    
    %% Método casualização da variável
    [nmarker nind] = size(entradasT);
    fprintf('\n');
    fprintf('Método : casualização da variável\n');
    imp= [];
    
    for j=1:nmarker
        
        xx = entradasT;
        embaralha = randperm(nind);
        xx(j,:) = xx(j,embaralha);
        yyrede = sim(minhanet,xx);
        matriz = [yyrede' saidasT'];
        erro = yyrede'- saidasT';
        EQM= mse(erro);
        [r,p] = corrcoef(matriz);
        R2 = r(1,2)*r(1,2);
        imp = [imp; EQM j];
    end   
   
    %%
     %______________________________________________
    % 8b. Salvando os resultados
    %______________________________________________
   colnames = {'EQM', 'marker'};
   imp = array2table(imp, 'VariableNames', colnames);
   
   names = {'nfold','method','variable'};
   addcol = table(repmat(i,height(imp),1),repmat('RBF',height(imp),1),repmat(varsel,height(imp),1), 'VariableNames',names);
   imp = [imp addcol];
   namesaida=['resultimprbfFOLD' num2str(i) '.xls'];
   writetable(imp, namesaida)
   
   if i == 1
            imprbf=imp;
   else
       imprbf=[imprbf;imp];
   end
    %%
   diary off
end
%%
%______________________________________________
% 9. Resultados Finais
%______________________________________________
resultrbf = [R2trei' R2val' reqt' reqv' raioo' nno'];
resultrbf = [resultrbf;
    mean(resultrbf);
    std(resultrbf)]
%%
colnames = {'R2Trein', 'R2Val', 'REQMTrein', 'REQMVal', 'RaioOtimo', 'NeuronioOtimo'};
resultrbf = array2table(resultrbf, 'VariableNames', colnames);
%%
rownames = {'KFold1','KFold2', 'KFold3','KFold4','KFold5','Mean', 'SD'};
names = {'nfold','method','variable'};
addcol = table( rownames',repmat('RBF',height(resultrbf),1), repmat(varsel,height(resultrbf),1),'VariableNames',names);
resultrbf = [resultrbf addcol]   

namesaida=['resultrbfVar' num2str(varsel) '.xls'];
writetable(resultrbf, namesaida)
%%
imprbf

namesaida=['resultimprbfVar' num2str(varsel) '.xls'];
writetable(imprbf, namesaida)
%%
if v ==1 
    resultrbft = resultrbf;
    imprbft = imprbf;
else
    resultrbft = [resultrbft; resultrbf];
    imprbft = [imprbft; imprbf];
    
end

end
writetable(resultrbft,  'resultrbf.xls');
writetable(imprbft,  'resultimprbf.xls');