path = strcat('C:\Users\Administrator\Desktop\ʵ��1\���ݼ�\���Լ�\image');
file = dir(fullfile(path,'*.png'));
for i =1:length(file)
    %img = imread(strcat('C:\Users\Administrator\Desktop\images\',file(i).name));
    %imwrite(img,strcat('C:\Users\Administrator\Desktop\images1\',num2str(i),'.png'));
    oldname = file(i).name;
    newname = strcat(oldname(1:length(oldname)-6),'.png');
    eval(['!rename' 32 oldname 32 newname]);

    
end