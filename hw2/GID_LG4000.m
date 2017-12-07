g = '/Users/liuziyu/Desktop/CS559/2008-03-11_13';
files = dir(g);
galleryFolders = files([files.isdir]);
n_gallery = length(galleryFolders);

p = '/Users/liuziyu/Desktop/CS559/2010-04-27_29 2';
files = dir(p);
probeFolders = files([files.isdir]);
n_probe = length(probeFolders);
hd_genuine = [];
hd_imposter = [];
G = n_gallery;
P = n_probe;
valueSet2 = cell(1,4*(P-2));
keySet2 = cell(1,4*(P-2));
for j = 3:P
    files_probe = fullfile(p, probeFolders(j).name, '*.tiff');
    files_probe = dir(files_probe);   
    
    imagepath2 = fullfile(p,probeFolders(j).name,files_probe(1).name);%probe left eye
    [template_left_probe, mask_left_probe] = createiristemplate(imagepath2);
    imagepath2 = fullfile(p,probeFolders(j).name,files_probe(13).name);%probe right eye
    [template_right_probe, mask_right_probe] = createiristemplate(imagepath2);
    
    valueSet2{4*(j-2)-3} = template_left_probe;
    valueSet2{4*(j-2)-2} = mask_left_probe;
    valueSet2{4*(j-2)-1} = template_right_probe;
    valueSet2{4*(j-2)} = mask_right_probe;   
    keySet2{4*(j-2)-3} = 4*(j-2)-3;
    keySet2{4*(j-2)-2} = 4*(j-2)-2;
    keySet2{4*(j-2)-1} = 4*(j-2)-1;
    keySet2{4*(j-2)} = 4*(j-2);   
end
probe = containers.Map(keySet2, valueSet2);

valueSet = cell(1,4*(G-2));
keySet = cell(1,4*(G-2));
for i = 3:G
    files_gallery = fullfile(g, galleryFolders(i).name, '*.tiff');
    files_gallery = dir(files_gallery);
    
    imagepath = fullfile(g,galleryFolders(i).name,files_gallery(2).name);%gallery right eye
    [template_right_gallery, mask_right_gallery] = createiristemplate(imagepath); 
    imagepath = fullfile(g,galleryFolders(i).name,files_gallery(7).name);%gallery left eye
    [template_left_gallery, mask_left_gallery] = createiristemplate(imagepath);
    
    valueSet{4*(i-2)-3} = template_right_gallery;
    valueSet{4*(i-2)-2} = mask_right_gallery;
    valueSet{4*(i-2)-1} = template_left_gallery;
    valueSet{4*(i-2)} = mask_left_gallery;
    keySet{4*(i-2)-3} = 4*(i-2)-3;
    keySet{4*(i-2)-2} = 4*(i-2)-2;
    keySet{4*(i-2)-1} = 4*(i-2)-1;
    keySet{4*(i-2)} = 4*(i-2);
end
gallery = containers.Map(keySet, valueSet);

for i = 3:G
    for j = 3:P
        template_right_gallery = gallery(4*(i-2)-3);
        mask_right_gallery = gallery(4*(i-2)-2);
        template_left_gallery = gallery(4*(i-2)-1);
        mask_left_gallery = gallery(4*(i-2));
        
        template_left_probe = probe(4*(j-2)-3);
        mask_left_probe = probe(4*(j-2)-2);
        template_right_probe = probe(4*(j-2)-1) ;
        mask_right_probe = probe(4*(j-2));   
        
        if probeFolders(j).name == galleryFolders(i).name  
           hd_genuine = [hd_genuine, gethammingdistance(template_right_gallery, mask_right_gallery, template_right_probe, mask_right_probe, 15)]; 
           hd_genuine = [hd_genuine, gethammingdistance(template_left_gallery, mask_left_gallery, template_left_probe, mask_left_probe, 15)]; 
           hd_imposter = [hd_imposter, gethammingdistance(template_right_gallery, mask_right_gallery, template_left_probe, mask_left_probe, 15)];
           hd_imposter = [hd_imposter, gethammingdistance(template_left_gallery, mask_left_gallery, template_right_probe, mask_right_probe, 15)];
        else  
           hd_imposter = [hd_imposter, gethammingdistance(template_right_gallery, mask_right_gallery, template_right_probe, mask_right_probe, 15)]; 
           hd_imposter = [hd_imposter, gethammingdistance(template_left_gallery, mask_left_gallery, template_left_probe, mask_left_probe, 15)]; 
           hd_imposter = [hd_imposter, gethammingdistance(template_right_gallery, mask_right_gallery, template_left_probe, mask_left_probe, 15)];
           hd_imposter = [hd_imposter, gethammingdistance(template_left_gallery, mask_left_gallery, template_right_probe, mask_right_probe, 15)];
        end    
    end
end