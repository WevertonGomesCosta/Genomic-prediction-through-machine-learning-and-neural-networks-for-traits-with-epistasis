function caminho =   trilha(genesdir)

fid = fopen([genesdir 'genes.def']);

C = textscan(fid, '%s%s');
fclose(fid);

username=getenv('USERNAME');
[nlin ncol] = size(C{1,1});

username = ['GenesUSER'];

for ii= 1:nlin
D = cell2mat(C{1,1}(ii,1));
tf = strcmp(D,username); 
    if tf == 1
    posi = ii + 1 ;   
end    
end



caminho = cell2mat(C{1,1}(posi,1));
end