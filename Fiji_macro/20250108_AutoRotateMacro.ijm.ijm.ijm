Bscan_tilt = 1.9
Y_tilt = -2.35

filePath = File.openDialog("Open .tif for realignment");
open(filePath);
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
run("Scale...", "x=1.0 y=2 z=1.0 width=256 height=256 depth=700 interpolation=Bilinear average process create");
selectImage(title);
close();

saveAs("Tiff", filePath);
close();