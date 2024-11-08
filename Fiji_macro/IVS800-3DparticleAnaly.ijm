run("Crop");
//run("Duplicate...", "title=mask");
run("Gaussian Blur...", "sigma=1 stack");
//run("Convert to Mask", "method=RenyiEntropy background=Dark calculate black");
//run("Threshold...");
//setThreshold(45, 255);
//setOption("BlackBackground", true);
run("3D Objects Counter", "threshold=65 slice=2 min.=50 max.=12574500 exclude_objects_on_edges objects surfaces statistics");

//run("Analyze Particles...", "size=15-450 circularity=0.10-1.00 show=[Overlay] display include");
saveAs("Results", "C:/Users/lzhu/Desktop/Results.csv");
close();
close();
selectImage("Data_3d_view.tif");
close();