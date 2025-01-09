//1. crop:X:250, Z:200
//2. "Scale...": "x=2 y=1 z=2 width=500 height=398 depth=500 interpolation=Bilinear average process create");
run("Gaussian Blur...", "sigma=2 stack");
run("3D Objects Counter", "threshold=52 slice=1 min.=50 max.=1721340000 exclude_objects_on_edges objects surfaces statistics");
//selectImage("Surface map of Data_3d_view-1.tif");
