Bscan_tilt = -0.0
Y_tilt = -4.45

if (nImages>0) 
	while (nImages>0) { 
	  selectImage(nImages); 
	  close(); } 
filePath = File.openDialog("Open .tif for realignment");
dir = File.getParent(filePath) + "\\";
//rotate log int file; 
logIntFile = "Data_3d_view.tif";
logIntFilePath = dir + logIntFile;
open(logIntFilePath);
title=getTitle();
run("Rotate... ", "angle=Bscan_tilt grid=1 interpolation=Bilinear stack");
run("Reslice [/]...", "output=1.000 start=Left rotate avoid");
selectImage(title);
close();
title=getTitle();
run("Rotate... ", "angle=Y_tilt grid=1 interpolation=Bilinear stack");
run("Reslice [/]...", "output=1.000 start=Top rotate avoid");
selectImage(title);
close();
title=getTitle();
run("Scale...", "x=1.0 y=1 z=1.0 width=256 height=256 depth=700 interpolation=Bilinear average process create");
selectImage(title);
close();
saveAs("Tiff", logIntFilePath);
close();
//rotate liv file; 
LivFile = "Data_IntImg_LIV.tif";
LivFilePath = dir + LivFile;
open(LivFilePath);
title=getTitle();
run("Rotate... ", "angle=Bscan_tilt grid=1 interpolation=Bilinear stack");
run("Reslice [/]...", "output=1.000 start=Left rotate avoid");
selectImage(title);
close();
title=getTitle();
run("Rotate... ", "angle=Y_tilt grid=1 interpolation=Bilinear stack");
run("Reslice [/]...", "output=1.000 start=Top rotate avoid");
selectImage(title);
close();
title=getTitle();
run("Scale...", "x=1.0 y=1 z=1.0 width=256 height=256 depth=700 interpolation=Bilinear average process create");
selectImage(title);
close();
saveAs("Tiff", LivFilePath);
close();
//rotate raw liv file; 
rawLivFile = "Data_IntImg_LIV_raw.tif";
rawLivFilePath = dir + rawLivFile;
open(rawLivFilePath);
title=getTitle();
run("Rotate... ", "angle=Bscan_tilt grid=1 interpolation=Bilinear stack");
run("Reslice [/]...", "output=1.000 start=Left rotate avoid");
selectImage(title);
close();
title=getTitle();
run("Rotate... ", "angle=Y_tilt grid=1 interpolation=Bilinear stack");
run("Reslice [/]...", "output=1.000 start=Top rotate avoid");
selectImage(title);
close();
title=getTitle();
run("Scale...", "x=1.0 y=1 z=1.0 width=256 height=256 depth=700 interpolation=Bilinear average process create");
selectImage(title);
close();
saveAs("Tiff", rawLivFilePath);
close();

open(LivFilePath);
run("Orthogonal Views");
