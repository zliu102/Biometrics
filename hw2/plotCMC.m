function plotCMC(probe, gal, name)
prob = load(probe);
gallery = load(gal);

[rowProb, colProb] = size(prob.veri);
[rowGal, colGal] = size(gallery.matrix);
maps = {};

%for i = 1 : rowProb
%    disp(i);
%   map = {};
%    name = prob.veri{i, 1};
%    eye = prob.veri{i, 4};
%    temp = prob.veri{i, 2};
%    mask = prob.veri{i, 3};
%    for j = 1 : rowGal
%        nameGal = gallery.matrix{j, 1};
%        eyeGal = gallery.matrix{j, 4};
%        tempGal = gallery.matrix{j, 2};
%        maskGal = gallery.matrix{j, 3};
%        hd = gethammingdistance(temp, mask, tempGal, maskGal, 1);
%        if isnan(hd)
%            continue;
%        end
%        map = [map; {nameGal, hd}];
%    end
%    [tempRow, tempCol] = size(map);
%    if tempRow == 0 || tempCol == 0
%        continue;
%    end
%    sorted = sortrows(map, [2]);
%    maps = [maps; {name, sorted}];
%end
%save(name,'maps');

maps = load(name);
maps = maps.maps;

[rowmap, colmap] = size(maps);
res = [];
for k = 1 : rowGal
    count = 0;
    for i = 1 : rowmap
        name = maps{i, 1};
        sortedMap = maps{i, 2};
        [row, col] = size(sortedMap);
        if k >= row
           count = count + 1;
           continue;
        end
        for j = 1 : k
            if strcmpi(sortedMap{j, 1}, name)
               count = count + 1;
               break;
            end
        end
    end
    count = count / rowmap;
    res = [res, count];
end
x = 1 : 1 : rowGal;
y = res;
plot(x, y);
xlabel('Rank');
ylabel('Recognition Rate');
