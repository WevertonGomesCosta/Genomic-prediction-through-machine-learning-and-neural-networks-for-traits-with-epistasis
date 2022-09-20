%% ************************************************************************
% ************************ 1. Inicialização *******************************
% *************************************************************************
clear,  close all, clc;
warning off
rand('seed', 1)

%% ************************************************************************
% ******************** 2. Leitura dos arquivos ****************************
% *************************************************************************
caminho = ('I:\Weverton\dados_qualificacao\Resultados\');
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
% *********** 4. Processamento da MPL otimizada *************************
% *************************************************************************

epocas = 500;
EQMi = 0.05;
ncamadas = 1;
EQMf = 0.05;
EQMpasso = 1;
funcoes = 'purelin';
Algo_tr = 'trainlm';
nvariavel = 18;
ninicial1 = 5;nfinal1 = 15;npasso1 = 2;
    v1i=1;v1f=1;v1p=1;
%%
for v = 1:nvariavel
% ************************************************************************
% *********** 3. Declaração dos parâmetros iniciais ***********************
% *************************************************************************
R2trein = zeros(1, 5);
R2val = zeros(1, 5);
reqt = zeros(1, 5);
reqv = zeros(1, 5);
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
    
    %% ************************************************************************
    % *********** 5. Montagem dos arquivos de entrada e saída *****************
    % *************************************************************************
    matrizfo = dadosTy(:,varsel)';
    saidasT = Normatiza(matrizfo,matrizfo,-1,1);
    
    matrizgo = dadosTx';
    entradasT  = Normatiza(matrizgo,matrizgo,-1,1);
    
    matrizfv = dadosVy;
    matrizfv = matrizfv(:,varsel)';
    saidasV  = Normatiza(matrizfv,matrizfo,-1,1);
    
    matrizgv= dadosVx';
    entradasV = Normatiza(matrizgv,matrizgo,-1,1);
    
    [nLin, nCol] = size(entradasT);
    numEntrada= nLin;
    numIndividuo= nCol;
    
    [nLinv, nColv] = size(entradasV);
    numEntradav= nLinv;
    numIndividuov= nColv;
    
    minimosmaximos = [-1*ones(numEntrada,1) ones(numEntrada,1)];
      
    %% ************************************************************************
    % *********** 6. Processamento da RNA   ***********************************
    % *************************************************************************
    R2tmin =0;
    R2vmin =0;
    R2t=-1;
    R2v=-1;
    perfvmin=999999999999999;
    
%%
    for EQMmin = EQMi:EQMpasso:EQMf
        if R2vmin >= 0.999 
            break
        end                  
                
    for k1 = ninicial1:npasso1:nfinal1 % Neuronios Camada 1
        if R2vmin >= 0.999 
            break
        end

            
    for v1 = v1i:v1p:v1f % Função de Ativação Camada 1
        if R2vmin >= 0.999  
            break
        end

            net = newff(minimosmaximos, [k1 1], { char(cellstr(funcoes(v1,:))) 'purelin' },  Algo_tr);

        net = init(net);
        %%
       
        %6a. ********************** CONFIGURANDO OS PARAMETROS DA REDE **********************
        net.trainParam.epochs = epocas; % NUMERO MAXIMO DE EPOCAS
        net.trainParam.goal = EQMmin; % CONDIÇAO DE PARADA POR ERRO
        net.trainParam.show = NaN; % INTERVALO PARA EXIBIÇAO NA TELA
        net.trainParam.max_fail=30;
        
        saidasRT = sim(net,entradasT);
        [net,tr]= train(net,entradasT,saidasT);
%%        
        %6b. *************** calculo do R2 treinamento e validação **************************
        matriz = [saidasT',saidasRT'];
        [r, ~] = corrcoef(matriz);
        R2t = r(1,2)*r(1,2);
        
        R2tmin = R2t;
        saidasRV = sim(net,entradasV);
        matriz = [saidasV',saidasRV'];
        [r, ~] = corrcoef(matriz);
        R2v = r(1,2)*r(1,2);
        perfV = perform(net,saidasV,saidasRV);
      
        if R2v >= R2vmin
            perfvmin = perfV;
            R2vmin = R2v;
            minhanet = net;
            EQMgol = EQMmin;
            
            %6c. ********************* resultado parcial do treinamento *********************
            k1f=k1; 
            v1ff=v1;
            fprintf('\n');
            fprintf('Neurônios na camada 1: %s\n',num2str(k1));
            fprintf('EQMgol: %s\n',num2str(EQMgol));
        
            if v1 > 0
                fprintf('Funçao Atv 1:   %s\n',char(cellstr(funcoes(v1,:))))
            end
            
            %%            
            fprintf('\n');
            fprintf('Treinamento\n');
            fprintf('Entradas: %5.0f\n',numEntrada);
            fprintf('Indivíduos: %5.0f\n',numIndividuo);
            fprintf('R2 (Porc.): %12.8f\n',  100*R2tmin);
            
            %7f. ************** resultado parcial da validação ***************************
            fprintf('\n');
            fprintf('Validação\n');
            fprintf('Entradas: %5.0f\n',numEntradav);
            fprintf('Indivíduos: %5.0f\n',numIndividuov);
            fprintf('R² (Porc.) : %12.8f\n',  100*R2vmin);
            %%
        end
    end
    end
    end


diary off

%% ************************************************************************
% *********** 7. Resultado Final  *****************************************
% *************************************************************************
diary on
diary([caminho 'final' num2str(varsel) num2str(i) '.txt'])

%______________________________________________
% 7a Topologia
%______________________________________________
fprintf('\n');
fprintf('------------------------------------   \n');
fprintf('Resultado FINAL   \n');
fprintf('------------------------------------   \n');
fprintf('Variável principal (Y) :  %d\n',varsel);
fprintf('Número do Fold :  %d\n',i);
fprintf('Épocas :  %d\n',epocas);
fprintf('EQM :  %d\n',EQMmin);
fprintf('EQMgol: %s\n',num2str(EQMgol));
fprintf('Camadas Ocultas :  %d\n',ncamadas);
fprintf('Neurônios na camada 1: %s\n',num2str(k1f));
fprintf('Treinamento :  %d\n',char(cellstr(Algo_tr)));
fprintf(' \n');
if v1ff > 0
    fprintf('Funçao Atv 1:   %s\n',char(cellstr(funcoes(v1ff,:))))
end

%______________________________________________
% 7b. Valores de R2 para treinamento e validação
%______________________________________________
fprintf('\n');
fprintf('Treinamento\n');
fprintf('Entradas: %5.0f\n',numEntrada);
fprintf('Indivíduos: %5.0f\n',numIndividuo);

saidasRT = sim(minhanet,entradasT);
matriz = [saidasT',saidasRT'];
[r, ~] = corrcoef(matriz);
R2trein(i) = r(1,2)*r(1,2);
y1  = DNormatiza(saidasRT ,matrizfo,-1,1);
y2  = DNormatiza(saidasT ,matrizfo,-1,1);
y1 = y1 - mean(y1) + mean(y2);
EQMtx= mse(y1'-y2');
reqt(i) = sqrt(EQMtx);

fprintf('\n');
fprintf('Validação\n');
fprintf('Entradas: %5.0f\n',numEntradav);
fprintf('Indivíduos: %5.0f\n',numIndividuov);
fprintf('\n');

saidasRV = sim(minhanet,entradasV);
matriz = [saidasV',saidasRV'];
[r, ~] = corrcoef(matriz);
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

resultmlp = [R2trein(i) R2val(i) reqt(i) reqv(i)];
rownames = {['KFold' num2str(i)]};
colnames = {'R2Trein', 'R2Val', 'REQMTrein', 'REQMVal'};   
resultmlp = array2table(resultmlp, 'VariableNames', colnames,'RowNames', rownames)

%______________________________________________
% 8 Importância das variáveis
%______________________________________________

yyrede = sim(minhanet,entradasT);
%______________________________________________
% 8a. Parâmetros iniciais 
%______________________________________________

matriz = [yyrede' saidasT'];
erro =yyrede'- saidasT';
EQMT= mse(erro);
[r,p] = corrcoef(matriz);
R2Ti = r(1,2)*r(1,2);

fprintf('Importância dos marcadores - Análise com todos caracteres\n');
fprintf('R2 porc.  :  %f\n',100*R2Ti);
fprintf('EQM  :  %f\n',EQMT);
fprintf('\n');

%% Método casualização da variável
[nmarker, nind] = size(entradasT);
fprintf('\n');
fprintf('Método : casualização da variável\n');

imp = [];

%%
for j=1:nmarker    %%
    xx = entradasT;
    embaralha = randperm(nind);
    xx(j,:) = xx(j,embaralha);
    yyrede = sim(minhanet,xx);
    matriz = [yyrede' saidasT'];
    erro = yyrede'- saidasT';
    EQM= mse(erro);
    [r,p] = corrcoef(matriz);
    R2 = r(1,2)*r(1,2);
    imp = [imp; EQM, j];
end
%%  
%______________________________________________
% 8b. Salvando os resultados
%______________________________________________
colnames = {'EQM', 'marker'};
imp = array2table(imp, 'VariableNames', colnames);
%%
names = {'nfold','method','variable'};
addcol = table(repmat(i,height(imp),1),repmat('MLP',height(imp),1),repmat(varsel,height(imp),1), 'VariableNames',names);
imp = [imp, addcol];
namesaida=['resultimpmlpFOLD' num2str(i) '.xls'];
writetable(imp, namesaida) 
%%
if i==1
    impmlp = imp;
else
    impmlp = [impmlp; imp];
end

%%
end
diary off
%%
%______________________________________________
% 9. Resultados Finais
%______________________________________________

resultmlp = [R2trein' R2val' reqt' reqv'];
resultmlp = [resultmlp;
    mean(resultmlp);
    std(resultmlp)]
%%
colnames = {'R2Trein', 'R2Val', 'REQMTrein', 'REQMVal'};
resultmlp = array2table(resultmlp, 'VariableNames', colnames)
%%
rownames = {'KFold1','KFold2', 'KFold3','KFold4','KFold5','Mean', 'SD'};
names = {'nfold','method','variable'};
addcol = table(rownames',repmat('MLP',height(resultmlp),1), repmat(varsel,height(resultmlp),1),'VariableNames',names);
resultmlp = [resultmlp addcol];   

namesaida=['resultmlpVar' num2str(varsel) '.xls'];
writetable(resultmlp, namesaida)
%%

namesaida=['resultimpmlpVar' num2str(varsel) '.xls']
writetable(impmlp, namesaida)
%%
if v == 1 
    resultmlpt = resultmlp;
    impmlpt = impmlp;
else
    resultmlpt = [resultmlpt; resultmlp];
    impmlpt = [impmlpt; impmlp];
end
end
writetable(resultmlpt, 'resultmlp.xls')
%%
save('resultimpmlp.mat', 'impmlpt')
%%
writetable(impmlpt, 'resultimpmlp.xls')
