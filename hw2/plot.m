% a = hd_genuine;
% b = hd_imposter;
a = hd_genuine;
b = hd_imposter;


% Histograms
h1 = histfit(a,40,'normal');
hold on;
h2 = histfit(b,40,'normal');

%// Get the lines here
hLines = findobj('Type','Line');
set(hLines(1),'Color','r')
set(hLines(2),'Color','g')

%delete(h1(1))
delete(h2(1))

hPatches=findobj(gca,'Type','patch');
set(hPatches,'FaceColor',[1 0 0],'EdgeColor','w')

hold off;

% Put up legend.
legend1 = sprintf('Genuine');
legend2 = sprintf('Genuine fitting curve');
legend3 = sprintf('Imposter');
legend4 = sprintf('Imposter fitting curve');
legend({legend1,legend2,legend3,legend4});