//filePath = File.openDialog("Open .tif for labell");
//dir = File.getParent(filePath) + "\\";
dir = "J:\\Data_2024\\20241113_BayerDemo\\Timelapse\\"
for (j=11; j<49; ++j){
//j = 9;
logIntFile = "Timelapse" + j + ".png";
logIntFilePath = dir + logIntFile;
open(logIntFilePath);
title=getTitle();

text = "" + j + "-hr";
setFont("SansSerif", 18, " antialiased");
setColor("white");
Overlay.drawString(text, 416, 81, 0.0);
Overlay.show();
run("Flatten");
selectImage(title);
close();
run("Save");
close();
}
