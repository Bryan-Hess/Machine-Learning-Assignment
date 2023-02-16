function dataPreProcessing()

%%Processes pgm files in source folders
delete ./input/train/*.pgm

for i = 1:40
    
  t1 = randi(10);
  t2 = randi(10);
  
  while(t1==t2)
    t2 = randi(10);
  end
  
  thisDir = strcat("input/all/s", string(i), '/');
  
  for j = 1:10
      
    thisTestDir = strcat("input/test/s", string(i), '/');
    
    if(j==1)
      delete(strcat(thisTestDir, '*'));
    end
    
    thisFolder = strcat(thisDir, string(j), ".pgm");
    
    if(j==t1 || j==t2)
        copyfile(thisFolder, thisTestDir);
    else
        copyfile(thisFolder, strcat("input/train/", string(i), '-', string(j), ".pgm"));
    end
    
  end

end